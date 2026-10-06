"""Regression checks for sweep integrity, inference and feature calculations."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import monk


class MonkTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
            dict(sample_id=sid, sweep_id=f'{sid}-{sweep}', elapsed_hours=hours,
                 temperature_c=temp, freq_hz=freq, magnitude=mag + hours,
                 phase=-10 - freq / 10000, data_source='synthetic')
            for sid in ('A', 'B')
            for sweep, hours, temp in ((1, 0, 24), (2, 20, 26))
            for freq, mag in ((10000, 400), (20000, 350), (30000, 300))
        ])

    def features(self, df=None, training=True):
        with contextlib.redirect_stdout(io.StringIO()):
            return monk.engineer_features(self.df if df is None else df, training)

    def test_one_row_per_sweep_and_no_target_features(self):
        X, y, groups, meta, columns = self.features()
        self.assertEqual(X.shape, (4, 12))
        self.assertNotIn('elapsed_hours', columns)
        self.assertEqual(len(meta), 4)
        self.assertEqual(groups.nunique(), 2)
        self.assertEqual(y.tolist(), [0, 20, 0, 20])

    def test_vectorized_slope_matches_polyfit(self):
        X, *_ = self.features()
        expected = np.polyfit(np.log10([10000, 20000, 30000]), np.log10([400, 350, 300]), 1)[0]
        self.assertAlmostEqual(X.mag_slope_loglog.iloc[0], expected, places=12)

    def test_prediction_without_known_age(self):
        X, y, _, meta, _ = self.features(self.df.drop(columns='elapsed_hours'), training=False)
        self.assertIsNone(y)
        self.assertNotIn('elapsed_hours', meta)
        self.assertEqual(len(X), 4)

    def test_legacy_csv_grouping(self):
        X, *_ = self.features(self.df.drop(columns='sweep_id'))
        self.assertEqual(len(X), 4)

    def test_duplicate_frequency_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate frequency'):
            self.features(pd.concat([self.df, self.df.iloc[[0]]]))

    def test_incomplete_sweep_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete sweep'):
            self.features(self.df.drop(index=0))

    def test_nonfinite_and_nonpositive_values_rejected(self):
        for value in (np.nan, np.inf, 0, -1):
            with self.subTest(value=value):
                df = self.df.astype({'magnitude': float}).copy()
                df.loc[0, 'magnitude'] = value
                with self.assertRaises(ValueError):
                    self.features(df)

    def test_inconsistent_metadata_rejected(self):
        for col in ('temperature_c', 'elapsed_hours'):
            df = self.df.copy()
            df.loc[0, col] += 1
            with self.assertRaisesRegex(ValueError, 'constant within'):
                self.features(df)

    def test_boundaries_and_saved_settings(self):
        self.assertEqual(list(monk.to_category([0, 16, 32, 48])), ['Fresh', 'Fresh', 'Moderate', 'Spoiled'])
        self.assertEqual(list(monk.to_category([2, 7, 12], [0, 5, 10, 15], ['A', 'B', 'C'])), ['A', 'B', 'C'])
        np.testing.assert_allclose(monk.to_freshness_score([0, 24, 48], 24), [100, 0, 0])

    def test_single_specimen_training_rejected(self):
        X, y, groups, *_ = self.features(self.df[self.df.sample_id == 'A'])
        with self.assertRaisesRegex(ValueError, 'at least two'):
            monk.train_evaluate(X, y, groups)

    def test_saved_model_prediction_and_html_escaping(self):
        X, y, _, _, cols = self.features()
        with tempfile.TemporaryDirectory() as directory, patch.object(monk, 'OUTPUT_DIR', Path(directory)), contextlib.redirect_stdout(io.StringIO()):
            model_path = Path(directory) / 'model.joblib'
            monk.save_model(X, y, 'Monk-LR', cols, model_path, 'synthetic')
            payload = monk.load_model(model_path)
            payload['max_hours'] = 24
            payload['cat_edges'] = [0, 5, 15, 24.01]
            import joblib
            joblib.dump(payload, model_path)
            unknown = self.df.drop(columns='elapsed_hours').copy()
            unknown.loc[unknown.sample_id == 'A', 'sample_id'] = '<script>alert(1)</script>'
            input_path = Path(directory) / 'UPPER.CSV'
            unknown.to_csv(input_path, index=False)
            out = monk.predict_from_csv(input_path, model_path)
            np.testing.assert_allclose(out.freshness, np.round(monk.to_freshness_score(out.pred_hours, 24), 1), atol=.3)
            self.assertTrue((Path(directory) / 'UPPER_predictions.csv').exists())
            self.assertEqual(pd.read_csv(input_path).shape, unknown.shape)
            report = (Path(directory) / 'monk_report.html').read_text()
            self.assertNotIn('<script>alert(1)</script>', report)
            self.assertIn('&lt;script&gt;', report)
            self.assertIn('synthetic', report)

    def test_prediction_frequency_mismatch(self):
        X, y, _, _, cols = self.features()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory)
            monk.save_model(X, y, 'Monk-LR', cols, path / 'model.joblib')
            df = self.df.copy()
            df.loc[df.freq_hz == 30000, 'freq_hz'] = 40000
            df.to_csv(path / 'input.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'Frequency mismatch'):
                monk.predict_from_csv(path / 'input.csv', path / 'model.joblib')


if __name__ == '__main__':
    unittest.main()
