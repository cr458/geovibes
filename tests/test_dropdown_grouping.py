"""Tests for dropdown grouping functionality.

These tests verify that database entries are properly grouped by region
for the dropdown UI.
"""


class TestGroupDatabasesByRegion:
    """Tests for grouping databases by region."""

    def test_groups_databases_by_region(self):
        """Should group databases by their region."""
        from geovibes.ui.app import group_databases_by_region

        databases = [
            {"db_path": "path1", "region": "alabama", "display_name": "DINO ViT"},
            {"db_path": "path2", "region": "alabama", "display_name": "EarthGenome"},
            {"db_path": "path3", "region": "new-mexico", "display_name": "Google Sat"},
        ]

        grouped = group_databases_by_region(databases)

        assert "alabama" in grouped
        assert "new-mexico" in grouped
        assert len(grouped["alabama"]) == 2
        assert len(grouped["new-mexico"]) == 1

    def test_handles_empty_list(self):
        """Should handle empty database list."""
        from geovibes.ui.app import group_databases_by_region

        grouped = group_databases_by_region([])
        assert grouped == {}

    def test_handles_missing_region(self):
        """Should use 'Other' for databases without region."""
        from geovibes.ui.app import group_databases_by_region

        databases = [
            {"db_path": "path1", "display_name": "Unknown DB"},
            {"db_path": "path2", "region": "alabama", "display_name": "Alabama DB"},
        ]

        grouped = group_databases_by_region(databases)

        assert "Other" in grouped
        assert len(grouped["Other"]) == 1
        assert grouped["Other"][0]["display_name"] == "Unknown DB"


class TestBuildGroupedDropdownItems:
    """Tests for building grouped dropdown items."""

    def test_builds_items_with_headers(self):
        """Should create items list with headers for each region."""
        from geovibes.ui.app import build_grouped_dropdown_items

        databases = [
            {
                "db_path": "path1",
                "region": "alabama",
                "display_name": "DINO ViT",
                "is_remote": True,
            },
            {
                "db_path": "path2",
                "region": "alabama",
                "display_name": "EarthGenome",
                "is_remote": False,
            },
        ]

        items = build_grouped_dropdown_items(databases)

        # Should have: header for alabama + 2 items
        assert len(items) == 3
        assert items[0] == {"header": "Alabama"}
        assert items[1]["value"] == "path1"
        assert items[2]["value"] == "path2"

    def test_adds_remote_prefix(self):
        """Should add Remote/ prefix for remote databases."""
        from geovibes.ui.app import build_grouped_dropdown_items

        databases = [
            {
                "db_path": "path1",
                "region": "alabama",
                "display_name": "DINO ViT",
                "is_remote": True,
            },
        ]

        items = build_grouped_dropdown_items(databases)

        # Skip header, check item text
        assert "Remote / DINO ViT" in items[1]["text"]

    def test_adds_local_prefix(self):
        """Should add Local/ prefix for local databases."""
        from geovibes.ui.app import build_grouped_dropdown_items

        databases = [
            {
                "db_path": "path1",
                "region": "alabama",
                "display_name": "Local DB",
                "is_remote": False,
            },
        ]

        items = build_grouped_dropdown_items(databases)

        # Skip header, check item text
        assert "Local / Local DB" in items[1]["text"]

    def test_adds_dividers_between_regions(self):
        """Should add dividers between region groups."""
        from geovibes.ui.app import build_grouped_dropdown_items

        databases = [
            {
                "db_path": "path1",
                "region": "alabama",
                "display_name": "DB1",
                "is_remote": True,
            },
            {
                "db_path": "path2",
                "region": "new-mexico",
                "display_name": "DB2",
                "is_remote": True,
            },
        ]

        items = build_grouped_dropdown_items(databases)

        # Should have: header1 + item1 + divider + header2 + item2
        assert len(items) == 5
        assert items[2] == {"divider": True}

    def test_includes_placeholder_when_not_connected(self):
        """Should include placeholder at start when not connected."""
        from geovibes.ui.app import build_grouped_dropdown_items

        databases = [
            {
                "db_path": "path1",
                "region": "alabama",
                "display_name": "DB1",
                "is_remote": True,
            },
        ]

        items = build_grouped_dropdown_items(databases, include_placeholder=True)

        assert items[0] == {
            "text": "Select a database...",
            "value": None,
            "disabled": True,
        }

    def test_region_name_capitalization(self):
        """Should capitalize region names in headers."""
        from geovibes.ui.app import build_grouped_dropdown_items

        databases = [
            {
                "db_path": "path1",
                "region": "new-mexico",
                "display_name": "DB1",
                "is_remote": True,
            },
        ]

        items = build_grouped_dropdown_items(databases)

        # Header should be capitalized properly
        assert items[0]["header"] == "New-Mexico"

    def test_sorts_regions_alphabetically(self):
        """Should sort regions alphabetically."""
        from geovibes.ui.app import build_grouped_dropdown_items

        databases = [
            {"db_path": "z", "region": "zulu", "display_name": "Z", "is_remote": True},
            {
                "db_path": "a",
                "region": "alabama",
                "display_name": "A",
                "is_remote": True,
            },
            {
                "db_path": "n",
                "region": "new-mexico",
                "display_name": "N",
                "is_remote": True,
            },
        ]

        items = build_grouped_dropdown_items(databases)

        # Find headers (skip dividers)
        headers = [item["header"] for item in items if "header" in item]
        assert headers == ["Alabama", "New-Mexico", "Zulu"]
