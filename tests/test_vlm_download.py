import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from task2_vlm.download import ranged_download


class Response(io.BytesIO):
    def __init__(self, payload, start, end, total, status=206):
        super().__init__(payload[start:end+1])
        self.status = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}"}


class DownloadTests(unittest.TestCase):
    def test_ranges_assemble_and_existing_pieces_resume_without_network(self):
        payload = b"0123456789abcdef"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            def source(request, timeout):
                start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
                calls.append((start, end))
                return Response(payload, start, end, len(payload))
            with patch("task2_vlm.download.urllib.request.urlopen", side_effect=source):
                result = ranged_download("https://example.invalid/public-data", root / "archive.zip",
                                         len(payload), '"etag"', workers=2, chunk_size=4)
            self.assertEqual(result.read_bytes(), payload)
            self.assertEqual(sorted(calls), [(0, 3), (4, 7), (8, 11), (12, 15)])
            # Do not delete the completed archive; test resume through a second
            # independent destination by preserving/copying its piece cache.
            second = root / "second.zip"
            import shutil
            shutil.copytree(root / "archive.zip.chunks", root / "second.zip.chunks")
            with patch("task2_vlm.download.urllib.request.urlopen", side_effect=AssertionError("No requests expected")):
                result = ranged_download("https://example.invalid/public-data", second,
                                         len(payload), '"etag"', workers=2, chunk_size=4)
            self.assertEqual(result.read_bytes(), payload)

    def test_wrong_server_range_never_publishes_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "archive.zip"
            def wrong(request, timeout):
                return Response(b"abcdef", 0, 5, 6, status=200)
            with patch("task2_vlm.download.urllib.request.urlopen", side_effect=wrong):
                with self.assertRaises(ValueError):
                    ranged_download("https://example.invalid/public-data", destination,
                                     6, '"etag"', workers=1, chunk_size=6)
            self.assertFalse(destination.exists())
            self.assertTrue(destination.with_suffix(".zip.chunks").is_dir())


if __name__ == "__main__":
    unittest.main()
