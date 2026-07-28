import unittest
import http.client
from pathlib import Path

from scripts.download_cic_iot_diad import (
    DatasetCrawler,
    download_with_retries,
    local_path_for_url,
    parse_registration_response,
)


class DownloadCicIotDiadTests(unittest.TestCase):
    def test_parse_registration_response_returns_redirect(self) -> None:
        redirect = parse_registration_response(
            b'{"ok": true, "redirect": "browse.php", "message": "Ready"}'
        )

        self.assertEqual(redirect, "browse.php")

    def test_parse_registration_response_reports_form_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "bad email"):
            parse_registration_response(
                b'{"ok": false, "message": "bad email", "type": "validation"}'
            )

    def test_discover_targets_keeps_dataset_files_and_internal_pages(self) -> None:
        crawler = DatasetCrawler(
            root_url="https://cicresearch.ca/IOTDataset/CIC-IoT-IDAD-Dataset-2024/",
            allowed_suffixes=(".csv", ".zip"),
        )
        html = """
        <a href="Monday.csv">csv</a>
        <a href="nested/Tuesday.zip?download=1">zip</a>
        <a href="download.php?file=nested%2FWednesday.csv">official csv download</a>
        <a href="download.php?file=README.txt">readme</a>
        <a href="browse.php?dir=nested">nested folder</a>
        <a href="../outside.csv">outside</a>
        <a href="ciclogo.jpg">logo</a>
        <a href="https://example.com/file.csv">external</a>
        """

        downloads, pages = crawler.discover_targets(
            html,
            "https://cicresearch.ca/IOTDataset/CIC-IoT-IDAD-Dataset-2024/browse.php",
        )

        self.assertEqual(
            downloads,
            [
                "https://cicresearch.ca/IOTDataset/CIC-IoT-IDAD-Dataset-2024/Monday.csv",
                "https://cicresearch.ca/IOTDataset/CIC-IoT-IDAD-Dataset-2024/nested/Tuesday.zip?download=1",
                "https://cicresearch.ca/IOTDataset/CIC-IoT-IDAD-Dataset-2024/download.php?file=nested%2FWednesday.csv",
            ],
        )
        self.assertEqual(
            pages,
            [
                "https://cicresearch.ca/IOTDataset/CIC-IoT-IDAD-Dataset-2024/browse.php?dir=nested",
            ],
        )

    def test_local_path_for_url_preserves_safe_relative_path(self) -> None:
        destination = Path("data/raw/cic_iot_diad_2024")

        path = local_path_for_url(
            "https://cicresearch.ca/IOTDataset/CIC-IoT-IDAD-Dataset-2024/nested/Tuesday.zip?download=1",
            destination,
            "https://cicresearch.ca/IOTDataset/CIC-IoT-IDAD-Dataset-2024/",
        )

        self.assertEqual(path, destination / "nested" / "Tuesday.zip")

    def test_local_path_for_download_php_uses_file_parameter(self) -> None:
        destination = Path("data/raw/cic_iot_diad_2024")

        path = local_path_for_url(
            "https://cicresearch.ca/IOTDataset/CIC-IoT-IDAD-Dataset-2024/download.php?file=Anomaly%2FBenign%2FFlow.csv",
            destination,
            "https://cicresearch.ca/IOTDataset/CIC-IoT-IDAD-Dataset-2024/",
        )

        self.assertEqual(path, destination / "Anomaly" / "Benign" / "Flow.csv")

    def test_discover_targets_can_filter_by_official_prefix(self) -> None:
        crawler = DatasetCrawler(
            root_url="https://cicresearch.ca/IOTDataset/CIC-IoT-IDAD-Dataset-2024/",
            allowed_suffixes=(".csv",),
            include_prefixes=("Anomaly Detection - Flow Based features",),
        )
        html = """
        <a href="browse.php?p=Anomaly+Detection+-+Flow+Based+features">flow folder</a>
        <a href="browse.php?p=Device+Identification_Anomaly+Detection+-+Packet+Based+Features">packet folder</a>
        <a href="download.php?file=Anomaly+Detection+-+Flow+Based+features%2FBenign%2Fa.csv">flow csv</a>
        <a href="download.php?file=Device+Identification_Anomaly+Detection+-+Packet+Based+Features%2FBenign%2Fb.csv">packet csv</a>
        """

        downloads, pages = crawler.discover_targets(
            html,
            "https://cicresearch.ca/IOTDataset/CIC-IoT-IDAD-Dataset-2024/browse.php",
        )

        self.assertEqual(
            downloads,
            [
                "https://cicresearch.ca/IOTDataset/CIC-IoT-IDAD-Dataset-2024/download.php?file=Anomaly+Detection+-+Flow+Based+features%2FBenign%2Fa.csv",
            ],
        )
        self.assertEqual(
            pages,
            [
                "https://cicresearch.ca/IOTDataset/CIC-IoT-IDAD-Dataset-2024/browse.php?p=Anomaly+Detection+-+Flow+Based+features",
            ],
        )

    def test_download_retries_wrap_incomplete_reads(self) -> None:
        class FailingOpener:
            def open(self, request, timeout):  # noqa: ANN001, ANN202
                raise http.client.IncompleteRead(b"partial")

        with self.assertRaisesRegex(RuntimeError, "could not download"):
            download_with_retries(
                FailingOpener(),
                ["https://cicresearch.ca/IOTDataset/CIC-IoT-IDAD-Dataset-2024/sample.csv"],
                Path("data/raw/cic_iot_diad_2024"),
                root_url="https://cicresearch.ca/IOTDataset/CIC-IoT-IDAD-Dataset-2024/",
                timeout=1,
                overwrite=False,
                retries=1,
            )


if __name__ == "__main__":
    unittest.main()
