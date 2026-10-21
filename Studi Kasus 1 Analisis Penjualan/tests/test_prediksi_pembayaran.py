import unittest
import pandas as pd
import os

class TestDataQuality(unittest.TestCase):
    def setUp(self):
        self.file_path = "data/invoice_data.csv"
        
    def test_file_exists(self):
        self.assertTrue(os.path.exists(self.file_path), "File dataset invoice_data.csv tidak ditemukan!")
        
    def test_no_missing_values(self):
        if os.path.exists(self.file_path):
            df = pd.read_csv(self.file_path)
            self.assertFalse(df.isnull().values.any(), "Terdapat missing values pada dataset!")
            
    def test_columns_exist(self):
        if os.path.exists(self.file_path):
            df = pd.read_csv(self.file_path)
            expected_cols = ['invoice_id', 'invoice_amount', 'payment_method', 'customer_history_late', 'status_payment']
            for col in expected_cols:
                self.assertIn(col, df.columns, f"Kolom {col} tidak ditemukan dalam dataset!")

if __name__ == '__main__':
    unittest.main()
