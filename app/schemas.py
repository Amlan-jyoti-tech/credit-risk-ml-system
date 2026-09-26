from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Optional, Literal
from datetime import datetime

class LoanApplicationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    # Core Loan Fields
    loan_amnt: float = Field(..., gt=0)
    term: Literal[' 36 months', ' 60 months', '36 months', '60 months']
    int_rate: float = Field(..., gt=0)
    installment: float = Field(..., gt=0)
    sub_grade: Literal['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 
                      'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 
                      'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 
                      'G1', 'G2', 'G3', 'G4', 'G5']
    home_ownership: Literal['MORTGAGE', 'RENT', 'OWN', 'OTHER', 'ANY', 'NONE']
    annual_inc: float = Field(..., ge=0)
    verification_status: Literal['Source Verified', 'Verified', 'Not Verified']
    dti: float = Field(..., ge=0)
    addr_state: Literal['CA', 'NY', 'TX', 'FL', 'IL', 'NJ', 'PA', 'OH', 'GA', 'VA', 
                       'NC', 'MI', 'MD', 'MA', 'WA', 'CO', 'IN', 'TN', 'AZ', 'MO', 
                       'MN', 'NV', 'SC', 'WI', 'OR', 'AL', 'CT', 'LA', 'UT', 'KY', 
                       'OK', 'KS', 'AR', 'HI', 'NM', 'WV', 'NH', 'RI', 'MS', 'NE', 
                       'MT', 'DE', 'AK', 'DC', 'WY', 'SD', 'VT', 'ME', 'ID', 'ND', 'IA']
    purpose: Literal['debt_consolidation', 'credit_card', 'home_improvement', 'other', 
                    'major_purchase', 'small_business', 'car', 'medical', 'moving', 
                    'vacation', 'house', 'wedding', 'renewable_energy', 'educational']
    application_type: Literal['Individual', 'Joint App']
    
    fico_range_low: float = Field(..., ge=300, le=850)
    fico_range_high: float = Field(..., ge=300, le=850)
    
    earliest_cr_line: str
    issue_d: str
    
    # Strictly Required Origination History Fields (Not imputed by FeatureEngineer)
    total_acc: float = Field(..., ge=0)
    open_acc: float = Field(..., ge=0)
    revol_bal: float = Field(..., ge=0)
    delinq_2yrs: float = Field(..., ge=0)
    acc_now_delinq: float = Field(..., ge=0)
    delinq_amnt: float = Field(..., ge=0)
    pub_rec: float = Field(..., ge=0)
    inq_last_6mths: float = Field(..., ge=0)
    
    # Optional Fields
    emp_length: Optional[Literal['< 1 year', '1 year', '2 years', '3 years', '4 years', 
                                '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']] = None
    initial_list_status: Optional[Literal['w', 'f']] = None
    
    revol_util: Optional[float] = Field(None, ge=0)
    tot_cur_bal: Optional[float] = Field(None, ge=0)
    total_rev_hi_lim: Optional[float] = Field(None, ge=0)
    tot_hi_cred_lim: Optional[float] = Field(None, ge=0)
    avg_cur_bal: Optional[float] = Field(None, ge=0)
    
    mths_since_last_delinq: Optional[float] = Field(None, ge=0)
    mths_since_last_record: Optional[float] = Field(None, ge=0)
    mths_since_last_major_derog: Optional[float] = Field(None, ge=0)
    mths_since_recent_bc_dlq: Optional[float] = Field(None, ge=0)
    mths_since_recent_revol_delinq: Optional[float] = Field(None, ge=0)
    
    pub_rec_bankruptcies: Optional[float] = Field(None, ge=0)
    tax_liens: Optional[float] = Field(None, ge=0)
    
    mths_since_recent_inq: Optional[float] = Field(None, ge=0)
    mort_acc: Optional[float] = Field(None, ge=0)
    num_bc_tl: Optional[float] = Field(None, ge=0)
    num_il_tl: Optional[float] = Field(None, ge=0)
    num_actv_bc_tl: Optional[float] = Field(None, ge=0)
    num_actv_rev_tl: Optional[float] = Field(None, ge=0)
    num_rev_accts: Optional[float] = Field(None, ge=0)
    num_tl_op_past_12m: Optional[float] = Field(None, ge=0)
    acc_open_past_24mths: Optional[float] = Field(None, ge=0)
    collections_12_mths_ex_med: Optional[float] = Field(None, ge=0)
    chargeoff_within_12_mths: Optional[float] = Field(None, ge=0)
    num_accts_ever_120_pd: Optional[float] = Field(None, ge=0)
    num_tl_90g_dpd_24m: Optional[float] = Field(None, ge=0)
    num_tl_30dpd: Optional[float] = Field(None, ge=0)
    num_tl_120dpd_2m: Optional[float] = Field(None, ge=0)
    pct_tl_nvr_dlq: Optional[float] = Field(None, ge=0)
    bc_util: Optional[float] = Field(None, ge=0)
    bc_open_to_buy: Optional[float] = Field(None, ge=0)
    percent_bc_gt_75: Optional[float] = Field(None, ge=0)
    total_bc_limit: Optional[float] = Field(None, ge=0)
    total_bal_ex_mort: Optional[float] = Field(None, ge=0)
    mo_sin_old_il_acct: Optional[float] = Field(None, ge=0)
    mo_sin_old_rev_tl_op: Optional[float] = Field(None, ge=0)
    mo_sin_rcnt_tl: Optional[float] = Field(None, ge=0)
    mths_since_recent_bc: Optional[float] = Field(None, ge=0)
    total_il_high_credit_limit: Optional[float] = Field(None, ge=0)
    num_rev_tl_bal_gt_0: Optional[float] = Field(None, ge=0)
    num_bc_sats: Optional[float] = Field(None, ge=0)
    tot_coll_amt: Optional[float] = Field(None, ge=0)

    @model_validator(mode='after')
    def validate_fico_range(self):
        if self.fico_range_low > self.fico_range_high:
            raise ValueError('fico_range_low must be less than or equal to fico_range_high')
        return self
        
    @model_validator(mode='after')
    def validate_dates(self):
        try:
            issue_date = datetime.strptime(self.issue_d, '%b-%Y')
            earliest_cr = datetime.strptime(self.earliest_cr_line, '%b-%Y')
        except ValueError as e:
            raise ValueError("Dates must be in 'Mon-YYYY' format (e.g., 'Jan-2005').")
            
        if earliest_cr > issue_date:
            raise ValueError('earliest_cr_line cannot be after issue_d')
            
        return self

class HealthResponse(BaseModel):
    status: str

class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    training_phase: str
    calibration_artifact: str
    feature_pipeline_version: str
    python_version: str
    scikit_learn_version: str

class PredictionResponse(BaseModel):
    raw_default_probability: float
    calibrated_default_probability: float
    model_name: str
    model_version: str
    calibration_method: str
    risk_band: str
    disclaimer: str = "ML engineering portfolio implementation. Not a production lending system."
