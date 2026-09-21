from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional
from datetime import datetime

class LoanApplicationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    # Required core origination fields
    loan_amnt: float = Field(..., gt=0)
    term: str
    int_rate: float = Field(..., gt=0)
    installment: float = Field(..., gt=0)
    sub_grade: str
    home_ownership: str
    annual_inc: float = Field(..., ge=0)
    verification_status: str
    dti: float = Field(..., ge=0)
    addr_state: str
    purpose: str
    application_type: str
    fico_range_low: float = Field(..., ge=300, le=850)
    fico_range_high: float = Field(..., ge=300, le=850)
    earliest_cr_line: str
    issue_d: str
    
    # Other origination fields
    initial_list_status: Optional[str] = None
    emp_length: Optional[str] = None
    
    total_acc: float = Field(..., ge=0)
    open_acc: float = Field(..., ge=0)
    revol_bal: float = Field(..., ge=0)
    revol_util: Optional[float] = None
    tot_cur_bal: Optional[float] = None
    total_rev_hi_lim: Optional[float] = None
    tot_hi_cred_lim: Optional[float] = None
    avg_cur_bal: Optional[float] = None
    
    delinq_2yrs: float = Field(..., ge=0)
    mths_since_last_delinq: Optional[float] = None
    mths_since_last_record: Optional[float] = None
    mths_since_last_major_derog: Optional[float] = None
    mths_since_recent_bc_dlq: Optional[float] = None
    mths_since_recent_revol_delinq: Optional[float] = None
    acc_now_delinq: float = Field(..., ge=0)
    delinq_amnt: float = Field(..., ge=0)
    
    pub_rec: float = Field(..., ge=0)
    pub_rec_bankruptcies: Optional[float] = None
    tax_liens: Optional[float] = None
    
    inq_last_6mths: float = Field(..., ge=0)
    mths_since_recent_inq: Optional[float] = None
    mort_acc: Optional[float] = None
    num_bc_tl: Optional[float] = None
    num_il_tl: Optional[float] = None
    num_actv_bc_tl: Optional[float] = None
    num_actv_rev_tl: Optional[float] = None
    num_rev_accts: Optional[float] = None
    num_tl_op_past_12m: Optional[float] = None
    acc_open_past_24mths: Optional[float] = None
    
    collections_12_mths_ex_med: Optional[float] = None
    chargeoff_within_12_mths: Optional[float] = None
    num_accts_ever_120_pd: Optional[float] = None
    num_tl_90g_dpd_24m: Optional[float] = None
    num_tl_30dpd: Optional[float] = None
    num_tl_120dpd_2m: Optional[float] = None
    pct_tl_nvr_dlq: Optional[float] = None
    
    tot_coll_amt: Optional[float] = None
    bc_util: Optional[float] = None
    bc_open_to_buy: Optional[float] = None
    percent_bc_gt_75: Optional[float] = None
    total_bc_limit: Optional[float] = None
    total_bal_ex_mort: Optional[float] = None
    
    mo_sin_old_il_acct: Optional[float] = None
    mo_sin_old_rev_tl_op: Optional[float] = None
    mo_sin_rcnt_tl: Optional[float] = None
    mths_since_recent_bc: Optional[float] = None
    total_il_high_credit_limit: Optional[float] = None
    num_rev_tl_bal_gt_0: Optional[float] = None
    num_bc_sats: Optional[float] = None

    @field_validator('issue_d', 'earliest_cr_line')
    def validate_dates(cls, v):
        try:
            datetime.strptime(v, '%b-%Y')
        except ValueError:
            raise ValueError("Dates must be in 'Mon-YYYY' format (e.g., 'Dec-2015')")
        return v
        
    @field_validator('fico_range_high')
    def validate_fico(cls, v, info):
        low = info.data.get('fico_range_low')
        if low is not None and v < low:
            raise ValueError("fico_range_high must be >= fico_range_low")
        return v


class PredictionResponse(BaseModel):
    raw_default_probability: float
    calibrated_default_probability: float
    model_name: str
    model_version: str
    calibration_method: str
    risk_band: str
    disclaimer: str = "ML engineering portfolio implementation. Not a production lending system."

class HealthResponse(BaseModel):
    status: str

class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    training_phase: str
    calibration_artifact: str
    feature_pipeline_version: str
