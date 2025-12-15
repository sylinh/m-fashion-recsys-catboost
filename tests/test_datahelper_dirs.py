import tempfile
import unittest
from pathlib import Path
from typing import List


from src.data.datahelper import DataHelper


class _ParquetStub:
    def __init__(self) -> None:
        self.writes: List[Path] = []

    def to_parquet(self, path: Path) -> None:
        self.writes.append(Path(path))
        Path(path).write_bytes(b"stub")


class TestDataHelperDirCreation(unittest.TestCase):
    def test_save_data_creates_nested_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            helper = DataHelper(data_dir=tmpdir)
            data = {"user": _ParquetStub(), "item": _ParquetStub(), "inter": _ParquetStub()}

            helper.save_data(data, name="processed_name")

            out_dir = Path(tmpdir) / "processed" / "processed_name"
            self.assertTrue(out_dir.is_dir())
            self.assertTrue((out_dir / "user.pqt").exists())
            self.assertTrue((out_dir / "item.pqt").exists())
            self.assertTrue((out_dir / "inter.pqt").exists())


if __name__ == "__main__":
    unittest.main()
