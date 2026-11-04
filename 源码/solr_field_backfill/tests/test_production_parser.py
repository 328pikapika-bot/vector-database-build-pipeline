"""Contract examples taken from build_milvus_db_all_v4 field parsing."""

import json

from solr_field_backfill.backfill import BackfillWorker, FIELD_SPECS, TARGET_FIELDS
from solr_field_backfill.production_parser import SOLR_QUERY_FIELDS, SOLR_SOURCES, parse_all


DOC_ID = "00000000000000000000000000000001"


def test_all_17_solr_sources_match_the_target_schema():
    assert set(SOLR_SOURCES) == set(TARGET_FIELDS) == set(FIELD_SPECS)
    assert SOLR_QUERY_FIELDS[0] == "id"
    assert len(SOLR_QUERY_FIELDS) == len(set(SOLR_SOURCES.values())) + 1


def test_reference_structured_and_scalar_rules_are_used_for_all_17_fields():
    source = ["S1∷title≡en|Journal A", "S2∷title≡Journal B"]
    files = ["F1∷sysFieldInfo≡fileType∷fileType≡pdf", "F2∷fileType≡pdf"]
    author = ["A1∷name≡张三"]
    organization = ["O1∷name≡机构"]
    doc = {
        "id": DOC_ID,
        "stringOM_hasSource": source,
        "stringOM_hasFile": files,
        "stringOM_hasAuthor": author,
        "stringOM_hasCreateOrganization": organization,
        "stringIMS_volume": ["en|7", "8"],
        "stringIMS_issue": ["en|2", "3"],
        "stringIS_startPage": ["10", "11"],
        "stringIS_endPage": ["20", "21"],
        "stringIS_pageCount": ["11", "12"],
        "dateIS_createTime": ["2020-01-01T00:00:00Z", "later"],
        "auto_stringITMS_publisher": ["en|Publisher", "Another"],
        "stringIMS_labelEvaluateGrade": ["en|YES", "NO"],
        "stringIS_sysProcessingUnit": ["CN311001", "CN311002"],
        "stringIMS_hasHoldinglibraryCode": ["en|CN311001", "CN311002"],
        "stringIMS_entityType": ["en|JournalPaper", "Book"],
    }
    expected = {
        "hasSource": "en|Journal A;Journal B",
        "hasFile": "pdf;pdf",
        "volume": "7",
        "issue": "2",
        "startPage": "10",
        "endPage": "20",
        "pageCount": "11",
        "createTime": "2020-01-01T00:00:00Z",
        "publisher": "en|Publisher",
        "labelEvaluateGrade": "YES",
        "sysProcessingUnit": "CN311001",
        "hasHoldinglibraryCode": "CN311001",
        "hasAuthor_str": json.dumps(author, ensure_ascii=False),
        "hasSource_str": json.dumps(source, ensure_ascii=False),
        "hasFile_str": json.dumps(files, ensure_ascii=False),
        "hasCreateOrganization_str": json.dumps(organization, ensure_ascii=False),
        "entityType": "JournalPaper",
    }
    assert parse_all(doc) == expected
    assert BackfillWorker.__new__(BackfillWorker).extract(doc) == {"id": DOC_ID, **expected}


def test_absent_values_stay_null_and_file_urls_are_not_treated_as_file_types():
    doc = {"id": DOC_ID, "stringOM_hasFile": ["http://example.org/document.pdf"],
           "stringIMS_entityType": ["", "Patent"]}
    parsed = parse_all(doc)
    assert parsed["hasFile"] is None
    assert parsed["hasFile_str"] == '["http://example.org/document.pdf"]'
    assert parsed["entityType"] is None
    assert all(value is None for key, value in parsed.items()
               if key not in {"hasFile_str"})
