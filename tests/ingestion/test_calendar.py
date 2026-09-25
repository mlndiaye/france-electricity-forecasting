from unittest.mock import Mock, patch

from felec.ingestion.calendar import fetch_jours_feries, fetch_vacances_scolaires

JOURS_FERIES_2024 = {
    "2024-01-01": "1er janvier",
    "2024-04-01": "Lundi de Pâques",
    "2024-05-01": "1er mai",
    "2024-05-08": "8 mai",
    "2024-05-09": "Ascension",
    "2024-05-20": "Lundi de Pentecôte",
    "2024-07-14": "14 juillet",
    "2024-08-15": "Assomption",
    "2024-11-01": "Toussaint",
    "2024-11-11": "11 novembre",
    "2024-12-25": "Jour de Noël",
}


@patch("felec.ingestion.calendar.requests.get")
def test_fetch_jours_feries_parses_response(mock_get):
    mock_get.return_value = Mock(json=lambda: JOURS_FERIES_2024, raise_for_status=lambda: None)

    result = fetch_jours_feries(2024)

    assert len(result) == 11
    assert set(result.columns) == {"date", "nom"}
    row = result[result["date"] == "2024-12-25"].iloc[0]
    assert row["nom"] == "Jour de Noël"


VACANCES_NOEL_RESPONSE = {
    "records": [
        {
            "fields": {
                "zones": "Zone A",
                "location": "Clermont-Ferrand",
                "start_date": "2023-12-22T23:00:00+00:00",
                "end_date": "2024-01-07T23:00:00+00:00",
                "description": "Vacances de Noël",
            }
        },
        {
            "fields": {
                "zones": "Zone A",
                "location": "Grenoble",
                "start_date": "2023-12-22T23:00:00+00:00",
                "end_date": "2024-01-07T23:00:00+00:00",
                "description": "Vacances de Noël",
            }
        },
        {
            "fields": {
                "zones": "Zone B",
                "location": "Lille",
                "start_date": "2023-12-22T23:00:00+00:00",
                "end_date": "2024-01-07T23:00:00+00:00",
                "description": "Vacances de Noël",
            }
        },
    ]
}


@patch("felec.ingestion.calendar.requests.get")
def test_fetch_vacances_scolaires_deduplicates_locations(mock_get):
    mock_get.return_value = Mock(
        json=lambda: VACANCES_NOEL_RESPONSE, raise_for_status=lambda: None
    )

    result = fetch_vacances_scolaires("2023-2024")

    # Two locations share the same Zone A period -> collapsed to one row.
    assert len(result) == 2
    assert set(result["zone"]) == {"Zone A", "Zone B"}
    zone_a = result[result["zone"] == "Zone A"].iloc[0]
    assert zone_a["date_debut"] == "2023-12-22"
    assert zone_a["date_fin"] == "2024-01-07"
