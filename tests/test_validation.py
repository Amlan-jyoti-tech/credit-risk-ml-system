import pytest
from pydantic import ValidationError
from app.schemas import LoanApplicationRequest

# Valid synthetic payload based on P0.10 rules
valid_payload = {
    "loan_amnt": 10000.0,
    "term": " 36 months",
    "int_rate": 10.99,
    "installment": 327.34,
    "sub_grade": "B4",
    "home_ownership": "MORTGAGE",
    "annual_inc": 75000.0,
    "verification_status": "Verified",
    "dti": 15.0,
    "addr_state": "CA",
    "purpose": "debt_consolidation",
    "application_type": "Individual",
    "fico_range_low": 700.0,
    "fico_range_high": 704.0,
    "earliest_cr_line": "Jan-2005",
    "issue_d": "Dec-2015",
    "total_acc": 20,
    "open_acc": 10,
    "revol_bal": 5000.0,
    "delinq_2yrs": 0.0,
    "acc_now_delinq": 0.0,
    "delinq_amnt": 0.0,
    "pub_rec": 0.0,
    "inq_last_6mths": 1.0
}

def test_valid_payload():
    request = LoanApplicationRequest(**valid_payload)
    assert request.loan_amnt == 10000.0
    
def test_leakage_rejection():
    # These fields must be rejected due to extra='forbid'
    leakage_fields = [
        "loan_status",
        "default_flag",
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_d",
        "last_pymnt_amnt",
        "next_pymnt_d",
        "last_fico_range_high",
        "last_fico_range_low",
        "last_credit_pull_d"
    ]
    
    for field in leakage_fields:
        payload = valid_payload.copy()
        payload[field] = "some_value"
        with pytest.raises(ValidationError) as exc_info:
            LoanApplicationRequest(**payload)
        assert "Extra inputs are not permitted" in str(exc_info.value)
        
def test_unknown_field_rejection():
    payload = valid_payload.copy()
    payload["some_random_unknown_field"] = 123
    with pytest.raises(ValidationError) as exc_info:
        LoanApplicationRequest(**payload)
    assert "Extra inputs are not permitted" in str(exc_info.value)

def test_invalid_fico_range():
    payload = valid_payload.copy()
    payload["fico_range_low"] = 750.0
    payload["fico_range_high"] = 700.0
    with pytest.raises(ValidationError) as exc_info:
        LoanApplicationRequest(**payload)
    assert "fico_range_low must be less than or equal to fico_range_high" in str(exc_info.value)

def test_fico_out_of_bounds():
    payload = valid_payload.copy()
    payload["fico_range_low"] = 250.0 # < 300
    with pytest.raises(ValidationError):
        LoanApplicationRequest(**payload)

def test_invalid_dates():
    payload = valid_payload.copy()
    payload["earliest_cr_line"] = "Dec-2016"
    payload["issue_d"] = "Jan-2015"
    with pytest.raises(ValidationError) as exc_info:
        LoanApplicationRequest(**payload)
    assert "earliest_cr_line cannot be after issue_d" in str(exc_info.value)
    
def test_invalid_date_format():
    payload = valid_payload.copy()
    payload["issue_d"] = "2015-12-01"
    with pytest.raises(ValidationError) as exc_info:
        LoanApplicationRequest(**payload)
    assert "Dates must be in 'Mon-YYYY' format" in str(exc_info.value)

def test_negative_loan_amount():
    payload = valid_payload.copy()
    payload["loan_amnt"] = -5000.0
    with pytest.raises(ValidationError):
        LoanApplicationRequest(**payload)

def test_negative_annual_inc():
    payload = valid_payload.copy()
    payload["annual_inc"] = -100.0
    with pytest.raises(ValidationError):
        LoanApplicationRequest(**payload)

def test_invalid_categorical():
    payload = valid_payload.copy()
    payload["term"] = "72 months"
    with pytest.raises(ValidationError):
        LoanApplicationRequest(**payload)
        
def test_optional_field_allowed():
    payload = valid_payload.copy()
    payload["emp_length"] = "5 years"
    payload["mths_since_last_delinq"] = 12.0
    request = LoanApplicationRequest(**payload)
    assert request.emp_length == "5 years"
    assert request.mths_since_last_delinq == 12.0
