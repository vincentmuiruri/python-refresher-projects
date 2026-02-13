# IFRS 9 ECL Model Assessment - Complete Framework
# Enhanced for Sidian Bank with XLSB support and multi-year processing
# IMPROVED VERSION - Enhanced with additional features and optimizations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from datetime import datetime, timedelta
import warnings
import sqlite3
import hashlib
from scipy import stats
from sklearn.preprocessing import StandardScaler
import json
import os
import glob
from pathlib import Path
import pyxlsb  # Required for reading xlsb files
from typing import Dict, List, Optional, Union, Tuple, Any
import re
from dataclasses import dataclass
from enum import Enum
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================================================================
# DATA QUALITY CONSTANTS AND ENUMS
# ===================================================================

class SeverityLevel(Enum):
    """Severity levels for data quality issues"""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    INFO = "Info"

class ValidationStatus(Enum):
    """Validation status"""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"

@dataclass
class DataQualityIssue:
    """Data structure for quality issues"""
    type: str
    field: str
    description: str
    severity: SeverityLevel
    records_affected: int
    percentage_affected: float
    recommendation: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

# ===================================================================
# CORE IFRS9 DATA QUALITY FRAMEWORK - IMPROVED VERSION
# ===================================================================

class IFRS9DataQualityFramework:
    """
    Comprehensive IFRS 9 data quality assessment framework
    Enhanced with performance optimizations, configurability, and additional checks
    """
    
    def __init__(self, df: pd.DataFrame, bank_name: str = "Bank_Data", 
                 config_file: Optional[str] = None):
        """
        Initialize the data quality framework
        
        Args:
            df: Input DataFrame
            bank_name: Name of the bank/institution
            config_file: Path to JSON configuration file
        """
        self.df = df.copy()
        self.bank_name = bank_name
        self.assessment_date = datetime.now()
        self.results = {}
        self.issues: List[DataQualityIssue] = []
        self.warnings: List[Dict] = []
        self.metrics = {}
        self.validation_history = []
        
        # Load configuration
        self.config = self._load_configuration(config_file)
        
        # Initialize field mapping
        self.field_mapping = {}
        self._initialize_field_mapping()
        
        # Track performance
        self.performance_metrics = {
            'start_time': datetime.now(),
            'operations': []
        }
    
    def _load_configuration(self, config_file: Optional[str]) -> Dict:
        """
        Load configuration from file or use defaults
        
        Returns:
            Configuration dictionary
        """
        default_config = {
            'thresholds': {
                'completeness_critical': 80.0,
                'completeness_warning': 95.0,
                'accuracy_critical': 95.0,
                'accuracy_warning': 98.0,
                'consistency_critical': 95.0,
                'consistency_warning': 99.0,
                'validity_critical': 90.0,
                'validity_warning': 95.0,
                'uniqueness_critical': 95.0,
                'uniqueness_warning': 99.0,
                'timeliness_critical': 90.0,
                'timeliness_warning': 95.0
            },
            'key_fields': {
                'customer_id': ['CUSTID', 'customer_id', 'cust_id', 'client_id', 'customer_number'],
                'account_id': ['AA.ID', 'account_id', 'acc_id', 'loan_id', 'account_number', 'contract_id'],
                'pd_value': ['SUSPD.AMT', 'pd', 'pd_value', 'probability_of_default', 'PD', 'PD_12M'],
                'lgd_value': ['lgd', 'lgd_value', 'loss_given_default', 'LGD', 'LGD_Value'],
                'ead_value': ['ead', 'ead_value', 'exposure_at_default', 'EAD', 'EAD_Value'],
                'stage': ['CLASS', 'stage', 'ifrs9_stage', 'classification', 'Stage', 'IFRS9_Stage'],
                'balance': ['C.LOAN.BAL', 'balance', 'outstanding_balance', 
                           'principal_balance', 'exposure', 'loan_amount'],
                'origination_date': ['START.DATE', 'origination_date', 'start_date', 
                                    'loan_date', 'disbursement_date', 'contract_date'],
                'maturity_date': ['MATURITY.DATE', 'maturity_date', 'end_date', 
                                 'due_date', 'final_payment_date'],
                'interest_rate': ['INT.RATE', 'interest_rate', 'rate', 'apr', 
                                 'interest_rate_pct', 'annual_rate'],
                'sector': ['SECTOR', 'industry', 'business_sector', 'naics_code'],
                'region': ['REGION', 'location', 'geography', 'branch_code'],
                'collateral_value': ['collateral_value', 'security_value', 'guarantee_amount'],
                'days_past_due': ['dpd', 'days_past_due', 'arrears_days', 'delinquency_days']
            },
            'validation_rules': {
                'pd_range': (0.0, 1.0),
                'lgd_range': (0.0, 1.0),
                'ead_min': 0.0,
                'balance_min': 0.0,
                'interest_rate_range': (0.0, 0.5),  # 0% to 50%
                'reasonable_years_range': (0, 30),  # For loan terms
                'max_dpd': 365  # Maximum days past due
            },
            'date_formats': [
                '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y%m%d', 
                '%d-%m-%Y', '%Y.%m.%d', '%d.%m.%Y', '%Y/%m/%d'
            ],
            'performance': {
                'enable_timing': True,
                'batch_size': 10000,
                'use_vectorization': True
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                # Merge user config with defaults
                self._merge_configs(default_config, user_config)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Failed to load config file: {e}. Using defaults.")
        
        return default_config
    
    def _merge_configs(self, default: Dict, user: Dict) -> None:
        """Recursively merge user configuration with defaults"""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_configs(default[key], value)
            else:
                default[key] = value
    
    def _initialize_field_mapping(self) -> None:
        """
        Initialize field mapping with improved matching algorithm
        """
        df_columns_lower = [col.lower().strip() for col in self.df.columns]
        column_patterns = {}
        
        # Create patterns for better matching
        for col in self.df.columns:
            col_lower = col.lower()
            # Remove special characters and split into words
            words = re.findall(r'\b\w+\b', col_lower)
            column_patterns[col] = {
                'full': col_lower,
                'words': words,
                'starts_with': [w[:3] for w in words if len(w) >= 3],
                'ends_with': [w[-3:] for w in words if len(w) >= 3]
            }
        
        for standard_name, possible_names in self.config['key_fields'].items():
            mapped = False
            
            # First pass: exact or partial matches
            for possible in possible_names:
                possible_lower = possible.lower()
                
                # Check for exact match
                if possible_lower in df_columns_lower:
                    idx = df_columns_lower.index(possible_lower)
                    self.field_mapping[standard_name] = self.df.columns[idx]
                    mapped = True
                    break
                
                # Check for partial matches
                for col, patterns in column_patterns.items():
                    if (possible_lower in patterns['full'] or
                        any(possible_lower in word for word in patterns['words']) or
                        any(possible_lower.startswith(start) for start in patterns['starts_with']) or
                        any(possible_lower.endswith(end) for end in patterns['ends_with'])):
                        self.field_mapping[standard_name] = col
                        mapped = True
                        break
                if mapped:
                    break
            
            # Second pass: use fuzzy matching if available
            if not mapped and hasattr(self, '_fuzzy_match'):
                best_match = self._fuzzy_match(standard_name, list(self.df.columns))
                if best_match:
                    self.field_mapping[standard_name] = best_match
                    mapped = True
            
            if not mapped:
                logger.warning(f"No match found for field: {standard_name}")
    
    def _time_operation(self, operation_name: str):
        """
        Decorator to time operations
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = datetime.now()
                result = func(*args, **kwargs)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                if self.config['performance']['enable_timing']:
                    self.performance_metrics['operations'].append({
                        'operation': operation_name,
                        'duration_seconds': duration,
                        'timestamp': start_time
                    })
                    logger.debug(f"Operation '{operation_name}' took {duration:.3f} seconds")
                
                return result
            return wrapper
        return decorator
    
    def run_comprehensive_assessment(self) -> Dict:
        """
        Run complete data quality assessment with enhanced features
        
        Returns:
            Comprehensive assessment results
        """
        logger.info(f"Starting IFRS 9 Data Quality Assessment for {self.bank_name}")
        print(f"\n{'='*80}")
        print(f"IFRS 9 DATA QUALITY ASSESSMENT")
        print(f"Bank: {self.bank_name}")
        print(f"Date: {self.assessment_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Dataset: {self.df.shape[0]} records, {self.df.shape[1]} columns")
        print(f"{'='*80}\n")
        
        # Record start time
        assessment_start = datetime.now()
        
        try:
            # Run all assessment categories with timing
            self.results['metadata'] = self._get_metadata()
            self.results['completeness'] = self._assess_completeness()
            self.results['accuracy'] = self._assess_accuracy()
            self.results['consistency'] = self._assess_consistency()
            self.results['validity'] = self._assess_validity()
            self.results['uniqueness'] = self._assess_uniqueness()
            self.results['timeliness'] = self._assess_timeliness()
            self.results['dimensionality'] = self._assess_dimensionality()
            self.results['distributions'] = self._assess_distributions()
            self.results['cross_validation'] = self._cross_field_validation()
            
            # Calculate scores and statistics
            self.results['overall_score'] = self._calculate_overall_score()
            self.results['risk_score'] = self._calculate_risk_score()
            self.results['summary_statistics'] = self._get_summary_statistics()
            self.results['issues_by_severity'] = self._categorize_issues()
            self.results['field_mapping'] = self.field_mapping
            
            # Performance metrics
            assessment_end = datetime.now()
            self.results['performance'] = {
                'total_duration_seconds': (assessment_end - assessment_start).total_seconds(),
                'operations': self.performance_metrics['operations'],
                'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
            }
            
            # Print comprehensive summary
            self._print_comprehensive_summary()
            
            # Generate recommendations
            self.results['recommendations'] = self._generate_recommendations()
            
            logger.info("Assessment completed successfully")
            
        except Exception as e:
            logger.error(f"Assessment failed: {e}")
            self.results['error'] = str(e)
            raise
        
        return self.results
    
    @_time_operation.__get__(object, type(None))
    def _get_metadata(self) -> Dict:
        """
        Get comprehensive dataset metadata
        """
        metadata = {
            'bank_name': self.bank_name,
            'assessment_date': self.assessment_date.isoformat(),
            'total_records': len(self.df),
            'total_columns': len(self.df.columns),
            'column_names': list(self.df.columns),
            'data_types': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            'field_mapping': self.field_mapping,
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'sample_data': self.df.head(3).to_dict('records') if len(self.df) > 0 else []
        }
        
        # Add column statistics
        metadata['column_statistics'] = {}
        for col in self.df.columns:
            metadata['column_statistics'][col] = {
                'null_count': self.df[col].isna().sum(),
                'null_percentage': (self.df[col].isna().sum() / len(self.df)) * 100,
                'unique_values': self.df[col].nunique(),
                'data_type': str(self.df[col].dtype)
            }
        
        return metadata
    
    @_time_operation.__get__(object, type(None))
    def _assess_completeness(self) -> List[Dict]:
        """
        Assess data completeness with enhanced checks
        """
        logger.info("Assessing Completeness...")
        completeness_results = []
        
        # Check all columns, not just key fields
        for col in self.df.columns:
            total_records = len(self.df)
            non_null_records = self.df[col].notna().sum()
            null_records = total_records - non_null_records
            completeness_pct = (non_null_records / total_records) * 100
            
            # Determine if this is a key field
            is_key_field = col in self.field_mapping.values()
            field_type = "Key Field" if is_key_field else "Standard Field"
            
            # Determine status based on thresholds
            thresholds = self.config['thresholds']
            
            if completeness_pct < thresholds['completeness_critical']:
                status = ValidationStatus.FAIL.value
                severity = SeverityLevel.CRITICAL
            elif completeness_pct < thresholds['completeness_warning']:
                status = ValidationStatus.WARNING.value
                severity = SeverityLevel.HIGH
            else:
                status = ValidationStatus.PASS.value
                severity = SeverityLevel.INFO
            
            # Add issue if not passing
            if status != ValidationStatus.PASS.value:
                issue = DataQualityIssue(
                    type='Completeness',
                    field=col,
                    description=f'Low completeness: {completeness_pct:.1f}%',
                    severity=severity,
                    records_affected=null_records,
                    percentage_affected=100 - completeness_pct,
                    recommendation='Consider data imputation or investigate data source'
                )
                self.issues.append(issue)
            
            completeness_results.append({
                'field': col,
                'field_type': field_type,
                'total_records': total_records,
                'non_null_records': non_null_records,
                'null_records': null_records,
                'completeness_percentage': round(completeness_pct, 2),
                'status': status,
                'severity': severity.value if status != ValidationStatus.PASS.value else None
            })
        
        # Overall completeness
        overall_completeness = self.df.notna().sum().sum() / (len(self.df) * len(self.df.columns)) * 100
        completeness_results.append({
            'field': 'OVERALL',
            'completeness_percentage': round(overall_completeness, 2),
            'status': ValidationStatus.PASS.value if overall_completeness >= 90 else ValidationStatus.FAIL.value
        })
        
        return completeness_results
    
    @_time_operation.__get__(object, type(None))
    def _assess_accuracy(self) -> List[Dict]:
        """
        Assess data accuracy with vectorized operations
        """
        logger.info("Assessing Accuracy...")
        accuracy_results = []
        
        # Define validation rules for different field types
        validation_rules = [
            # PD validation
            {
                'field_key': 'pd_value',
                'test_name': 'PD_Range_Validation',
                'validation_func': lambda x: ((x >= 0) & (x <= 1)),
                'error_message': 'PD value outside [0,1] range'
            },
            # LGD validation
            {
                'field_key': 'lgd_value',
                'test_name': 'LGD_Range_Validation',
                'validation_func': lambda x: ((x >= 0) & (x <= 1)),
                'error_message': 'LGD value outside [0,1] range'
            },
            # Balance validation
            {
                'field_key': 'balance',
                'test_name': 'Balance_Positive_Check',
                'validation_func': lambda x: (x >= 0),
                'error_message': 'Negative balance found'
            },
            # EAD validation
            {
                'field_key': 'ead_value',
                'test_name': 'EAD_Positive_Check',
                'validation_func': lambda x: (x >= 0),
                'error_message': 'Negative EAD found'
            },
            # Days Past Due validation
            {
                'field_key': 'days_past_due',
                'test_name': 'DPD_Range_Check',
                'validation_func': lambda x: ((x >= 0) & (x <= self.config['validation_rules']['max_dpd'])),
                'error_message': 'DPD outside valid range'
            }
        ]
        
        for rule in validation_rules:
            field_key = rule['field_key']
            if field_key in self.field_mapping:
                col = self.field_mapping[field_key]
                if col in self.df.columns:
                    # Convert to numeric if needed
                    series = pd.to_numeric(self.df[col], errors='coerce')
                    valid_mask = rule['validation_func'](series)
                    
                    valid_records = valid_mask.sum()
                    total_records = series.notna().sum()
                    accuracy_rate = (valid_records / total_records * 100) if total_records > 0 else 0
                    
                    # Determine status
                    thresholds = self.config['thresholds']
                    if accuracy_rate < thresholds['accuracy_critical']:
                        status = ValidationStatus.FAIL.value
                        severity = SeverityLevel.CRITICAL
                    elif accuracy_rate < thresholds['accuracy_warning']:
                        status = ValidationStatus.WARNING.value
                        severity = SeverityLevel.HIGH
                    else:
                        status = ValidationStatus.PASS.value
                        severity = SeverityLevel.INFO
                    
                    # Add issue if needed
                    if status != ValidationStatus.PASS.value:
                        issue = DataQualityIssue(
                            type='Accuracy',
                            field=col,
                            description=f'{rule["error_message"]}: {100-accuracy_rate:.1f}% invalid',
                            severity=severity,
                            records_affected=total_records - valid_records,
                            percentage_affected=100 - accuracy_rate,
                            recommendation='Review data source and validation rules'
                        )
                        self.issues.append(issue)
                    
                    accuracy_results.append({
                        'test': rule['test_name'],
                        'field': col,
                        'valid_records': int(valid_records),
                        'total_records': int(total_records),
                        'invalid_records': int(total_records - valid_records),
                        'accuracy_rate': round(accuracy_rate, 2),
                        'status': status,
                        'severity': severity.value if status != ValidationStatus.PASS.value else None
                    })
        
        return accuracy_results
    
    @_time_operation.__get__(object, type(None))
    def _assess_consistency(self) -> List[Dict]:
        """
        Assess data consistency with improved checks
        """
        logger.info("Assessing Consistency...")
        consistency_results = []
        
        # Stage consistency
        if 'stage' in self.field_mapping:
            stage_col = self.field_mapping['stage']
            if stage_col in self.df.columns:
                # Handle various stage formats
                valid_stages = {1, 2, 3, '1', '2', '3', 'Stage 1', 'Stage 2', 'Stage 3', 
                               'stage1', 'stage2', 'stage3', 'S1', 'S2', 'S3'}
                
                # Convert to string for comparison
                stage_series = self.df[stage_col].astype(str).str.strip()
                valid_mask = stage_series.isin(valid_stages)
                
                valid_records = valid_mask.sum()
                total_records = self.df[stage_col].notna().sum()
                consistency_rate = (valid_records / total_records * 100) if total_records > 0 else 0
                
                # Determine status
                thresholds = self.config['thresholds']
                if consistency_rate < thresholds['consistency_critical']:
                    status = ValidationStatus.FAIL.value
                    severity = SeverityLevel.CRITICAL
                elif consistency_rate < thresholds['consistency_warning']:
                    status = ValidationStatus.WARNING.value
                    severity = SeverityLevel.HIGH
                else:
                    status = ValidationStatus.PASS.value
                    severity = SeverityLevel.INFO
                
                if status != ValidationStatus.PASS.value:
                    issue = DataQualityIssue(
                        type='Consistency',
                        field=stage_col,
                        description=f'Invalid stage values: {100-consistency_rate:.1f}%',
                        severity=severity,
                        records_affected=total_records - valid_records,
                        percentage_affected=100 - consistency_rate,
                        recommendation='Stage values must be 1, 2, or 3 (or equivalent)'
                    )
                    self.issues.append(issue)
                
                consistency_results.append({
                    'test': 'Stage_Values_Consistency',
                    'field': stage_col,
                    'valid_records': int(valid_records),
                    'total_records': int(total_records),
                    'consistency_rate': round(consistency_rate, 2),
                    'status': status,
                    'severity': severity.value if status != ValidationStatus.PASS.value else None,
                    'valid_values_found': stage_series[valid_mask].unique().tolist()[:10]  # First 10 unique valid values
                })
        
        # Date consistency
        orig_col = self.field_mapping.get('origination_date')
        mat_col = self.field_mapping.get('maturity_date')
        
        if orig_col and mat_col and orig_col in self.df.columns and mat_col in self.df.columns:
            try:
                orig_dates = self._parse_date_column(orig_col)
                mat_dates = self._parse_date_column(mat_col)
                
                if orig_dates is not None and mat_dates is not None:
                    valid_mask = (orig_dates < mat_dates) & orig_dates.notna() & mat_dates.notna()
                    total_pairs = (orig_dates.notna() & mat_dates.notna()).sum()
                    consistency_rate = (valid_mask.sum() / total_pairs * 100) if total_pairs > 0 else 0
                    
                    # Determine status
                    thresholds = self.config['thresholds']
                    if consistency_rate < thresholds['consistency_critical']:
                        status = ValidationStatus.FAIL.value
                        severity = SeverityLevel.CRITICAL
                    elif consistency_rate < thresholds['consistency_warning']:
                        status = ValidationStatus.WARNING.value
                        severity = SeverityLevel.HIGH
                    else:
                        status = ValidationStatus.PASS.value
                        severity = SeverityLevel.INFO
                    
                    if status != ValidationStatus.PASS.value:
                        issue = DataQualityIssue(
                            type='Consistency',
                            field=f'{orig_col}, {mat_col}',
                            description=f'Date order inconsistency: {100-consistency_rate:.1f}%',
                            severity=severity,
                            records_affected=total_pairs - valid_mask.sum(),
                            percentage_affected=100 - consistency_rate,
                            recommendation='Origination date must be before maturity date'
                        )
                        self.issues.append(issue)
                    
                    consistency_results.append({
                        'test': 'Date_Order_Consistency',
                        'fields': f'{orig_col}, {mat_col}',
                        'valid_records': int(valid_mask.sum()),
                        'total_records': int(total_pairs),
                        'consistency_rate': round(consistency_rate, 2),
                        'status': status,
                        'severity': severity.value if status != ValidationStatus.PASS.value else None,
                        'avg_loan_term_days': ((mat_dates - orig_dates).dt.days.mean() 
                                              if not ((mat_dates - orig_dates).dt.days.isna().all()) 
                                              else None)
                    })
            except Exception as e:
                logger.error(f"Date consistency check failed: {e}")
                consistency_results.append({
                    'test': 'Date_Order_Consistency',
                    'fields': f'{orig_col}, {mat_col}',
                    'status': ValidationStatus.ERROR.value,
                    'error': str(e)
                })
        
        # Business rule consistency: EAD should not exceed balance significantly
        ead_col = self.field_mapping.get('ead_value')
        bal_col = self.field_mapping.get('balance')
        
        if ead_col and bal_col and ead_col in self.df.columns and bal_col in self.df.columns:
            try:
                ead_values = pd.to_numeric(self.df[ead_col], errors='coerce')
                bal_values = pd.to_numeric(self.df[bal_col], errors='coerce')
                
                # EAD should not exceed balance by more than 10%
                valid_mask = (ead_values <= bal_values * 1.1) & ead_values.notna() & bal_values.notna()
                total_pairs = (ead_values.notna() & bal_values.notna()).sum()
                consistency_rate = (valid_mask.sum() / total_pairs * 100) if total_pairs > 0 else 0
                
                if consistency_rate < 95:
                    issue = DataQualityIssue(
                        type='Consistency',
                        field=f'{ead_col}, {bal_col}',
                        description=f'EAD exceeds balance significantly: {100-consistency_rate:.1f}%',
                        severity=SeverityLevel.MEDIUM,
                        records_affected=total_pairs - valid_mask.sum(),
                        percentage_affected=100 - consistency_rate,
                        recommendation='Review EAD calculation methodology'
                    )
                    self.issues.append(issue)
                
                consistency_results.append({
                    'test': 'EAD_Balance_Consistency',
                    'fields': f'{ead_col}, {bal_col}',
                    'valid_records': int(valid_mask.sum()),
                    'total_records': int(total_pairs),
                    'consistency_rate': round(consistency_rate, 2),
                    'status': ValidationStatus.PASS.value if consistency_rate >= 95 else ValidationStatus.WARNING.value
                })
            except Exception as e:
                logger.error(f"EAD-Balance consistency check failed: {e}")
        
        return consistency_results
    
    @_time_operation.__get__(object, type(None))
    def _assess_validity(self) -> List[Dict]:
        """
        Assess data validity
        """
        logger.info("Assessing Validity...")
        validity_results = []
        
        # Customer ID validity
        if 'customer_id' in self.field_mapping:
            cust_col = self.field_mapping['customer_id']
            if cust_col in self.df.columns:
                # Check for valid format (not null, reasonable length, no special patterns)
                cust_series = self.df[cust_col].astype(str)
                
                # Define validity criteria
                length_valid = cust_series.str.len().between(3, 50)
                pattern_valid = ~cust_series.str.contains(r'[^\w\-]', na=False)  # Alphanumeric and hyphens only
                not_test = ~cust_series.str.lower().str.contains('test|demo|sample', na=False)
                
                valid_mask = length_valid & pattern_valid & not_test & cust_series.notna()
                validity_rate = (valid_mask.sum() / len(self.df) * 100)
                
                thresholds = self.config['thresholds']
                if validity_rate < thresholds['validity_critical']:
                    status = ValidationStatus.FAIL.value
                    severity = SeverityLevel.CRITICAL
                elif validity_rate < thresholds['validity_warning']:
                    status = ValidationStatus.WARNING.value
                    severity = SeverityLevel.HIGH
                else:
                    status = ValidationStatus.PASS.value
                    severity = SeverityLevel.INFO
                
                if status != ValidationStatus.PASS.value:
                    issue = DataQualityIssue(
                        type='Validity',
                        field=cust_col,
                        description=f'Invalid customer IDs: {100-validity_rate:.1f}%',
                        severity=severity,
                        records_affected=len(self.df) - valid_mask.sum(),
                        percentage_affected=100 - validity_rate,
                        recommendation='Implement customer ID validation rules'
                    )
                    self.issues.append(issue)
                
                validity_results.append({
                    'test': 'Customer_ID_Validity',
                    'field': cust_col,
                    'valid_records': int(valid_mask.sum()),
                    'total_records': len(self.df),
                    'validity_rate': round(validity_rate, 2),
                    'status': status,
                    'severity': severity.value if status != ValidationStatus.PASS.value else None,
                    'invalid_patterns_found': cust_series[~valid_mask].value_counts().head(5).to_dict()
                })
        
        # Interest rate validity
        if 'interest_rate' in self.field_mapping:
            rate_col = self.field_mapping['interest_rate']
            if rate_col in self.df.columns:
                try:
                    rates = pd.to_numeric(self.df[rate_col], errors='coerce')
                    # Handle percentages (if > 1, assume percentage)
                    rates = rates.apply(lambda x: x/100 if pd.notna(x) and abs(x) > 1 else x)
                    
                    min_rate, max_rate = self.config['validation_rules']['interest_rate_range']
                    valid_mask = (rates >= min_rate) & (rates <= max_rate) & rates.notna()
                    validity_rate = (valid_mask.sum() / rates.notna().sum() * 100) if rates.notna().sum() > 0 else 0
                    
                    thresholds = self.config['thresholds']
                    if validity_rate < thresholds['validity_critical']:
                        status = ValidationStatus.FAIL.value
                        severity = SeverityLevel.CRITICAL
                    elif validity_rate < thresholds['validity_warning']:
                        status = ValidationStatus.WARNING.value
                        severity = SeverityLevel.HIGH
                    else:
                        status = ValidationStatus.PASS.value
                        severity = SeverityLevel.INFO
                    
                    if status != ValidationStatus.PASS.value:
                        issue = DataQualityIssue(
                            type='Validity',
                            field=rate_col,
                            description=f'Invalid interest rates: {100-validity_rate:.1f}% outside [{min_rate*100}%, {max_rate*100}%]',
                            severity=severity,
                            records_affected=rates.notna().sum() - valid_mask.sum(),
                            percentage_affected=100 - validity_rate,
                            recommendation='Check interest rate calculation and input validation'
                        )
                        self.issues.append(issue)
                    
                    validity_results.append({
                        'test': 'Interest_Rate_Validity',
                        'field': rate_col,
                        'valid_records': int(valid_mask.sum()),
                        'total_records': int(rates.notna().sum()),
                        'validity_rate': round(validity_rate, 2),
                        'status': status,
                        'severity': severity.value if status != ValidationStatus.PASS.value else None,
                        'range_checked': f'{min_rate*100}% to {max_rate*100}%'
                    })
                except Exception as e:
                    logger.error(f"Interest rate validity check failed: {e}")
        
        return validity_results
    
    @_time_operation.__get__(object, type(None))
    def _assess_uniqueness(self) -> List[Dict]:
        """
        Assess data uniqueness
        """
        logger.info("Assessing Uniqueness...")
        uniqueness_results = []
        
        # Check for duplicate customer-account combinations
        cust_col = self.field_mapping.get('customer_id')
        acc_col = self.field_mapping.get('account_id')
        
        if cust_col and acc_col and cust_col in self.df.columns and acc_col in self.df.columns:
            total_records = len(self.df)
            unique_combinations = self.df[[cust_col, acc_col]].drop_duplicates().shape[0]
            duplicates_count = total_records - unique_combinations
            uniqueness_rate = (unique_combinations / total_records * 100)
            
            thresholds = self.config['thresholds']
            if uniqueness_rate < thresholds['uniqueness_critical']:
                status = ValidationStatus.FAIL.value
                severity = SeverityLevel.CRITICAL
            elif uniqueness_rate < thresholds['uniqueness_warning']:
                status = ValidationStatus.WARNING.value
                severity = SeverityLevel.HIGH
            else:
                status = ValidationStatus.PASS.value
                severity = SeverityLevel.INFO
            
            if status != ValidationStatus.PASS.value:
                issue = DataQualityIssue(
                    type='Uniqueness',
                    field=f'{cust_col}, {acc_col}',
                    description=f'Duplicate records found: {duplicates_count} duplicates ({100-uniqueness_rate:.1f}%)',
                    severity=severity,
                    records_affected=duplicates_count,
                    percentage_affected=100 - uniqueness_rate,
                    recommendation='Investigate duplicate records and implement deduplication process'
                )
                self.issues.append(issue)
            
            uniqueness_results.append({
                'test': 'Customer_Account_Uniqueness',
                'fields': f'{cust_col}, {acc_col}',
                'total_records': total_records,
                'unique_records': unique_combinations,
                'duplicate_records': duplicates_count,
                'uniqueness_rate': round(uniqueness_rate, 2),
                'status': status,
                'severity': severity.value if status != ValidationStatus.PASS.value else None
            })
        
        # Check for duplicate account IDs
        if 'account_id' in self.field_mapping:
            acc_col = self.field_mapping['account_id']
            if acc_col in self.df.columns:
                duplicate_accounts = self.df[acc_col].duplicated().sum()
                uniqueness_rate = ((len(self.df) - duplicate_accounts) / len(self.df)) * 100
                
                if uniqueness_rate < 99.9:
                    issue = DataQualityIssue(
                        type='Uniqueness',
                        field=acc_col,
                        description=f'Duplicate account IDs: {duplicate_accounts} duplicates',
                        severity=SeverityLevel.HIGH,
                        records_affected=duplicate_accounts,
                        percentage_affected=100 - uniqueness_rate,
                        recommendation='Account IDs should be unique. Investigate system integration issues.'
                    )
                    self.issues.append(issue)
                
                uniqueness_results.append({
                    'test': 'Account_ID_Uniqueness',
                    'field': acc_col,
                    'total_records': len(self.df),
                    'unique_accounts': self.df[acc_col].nunique(),
                    'duplicate_accounts': int(duplicate_accounts),
                    'uniqueness_rate': round(uniqueness_rate, 2),
                    'status': ValidationStatus.PASS.value if uniqueness_rate >= 99.9 else ValidationStatus.FAIL.value
                })
        
        return uniqueness_results
    
    @_time_operation.__get__(object, type(None))
    def _assess_timeliness(self) -> List[Dict]:
        """
        Assess data timeliness
        """
        logger.info("Assessing Timeliness...")
        timeliness_results = []
        
        # Check for recent origination dates
        if 'origination_date' in self.field_mapping:
            orig_col = self.field_mapping['origination_date']
            if orig_col in self.df.columns:
                try:
                    orig_dates = self._parse_date_column(orig_col)
                    if orig_dates is not None:
                        current_date = datetime.now()
                        
                        # Reasonable range: not more than 30 years ago, not in future
                        min_date = current_date - timedelta(days=30*365)
                        max_date = current_date + timedelta(days=30)  # Allow slight future dates
                        
                        valid_mask = (orig_dates >= min_date) & (orig_dates <= max_date) & orig_dates.notna()
                        total_dates = orig_dates.notna().sum()
                        timeliness_rate = (valid_mask.sum() / total_dates * 100) if total_dates > 0 else 0
                        
                        # Check for future dates (more serious issue)
                        future_dates = (orig_dates > current_date + timedelta(days=7)).sum()
                        
                        thresholds = self.config['thresholds']
                        if timeliness_rate < thresholds['timeliness_critical']:
                            status = ValidationStatus.FAIL.value
                            severity = SeverityLevel.CRITICAL
                        elif timeliness_rate < thresholds['timeliness_warning']:
                            status = ValidationStatus.WARNING.value
                            severity = SeverityLevel.HIGH
                        else:
                            status = ValidationStatus.PASS.value
                            severity = SeverityLevel.INFO
                        
                        if status != ValidationStatus.PASS.value:
                            issue_desc = f'Dates outside reasonable range: {100-timeliness_rate:.1f}%'
                            if future_dates > 0:
                                issue_desc += f' ({future_dates} future dates)'
                            
                            issue = DataQualityIssue(
                                type='Timeliness',
                                field=orig_col,
                                description=issue_desc,
                                severity=severity,
                                records_affected=total_dates - valid_mask.sum(),
                                percentage_affected=100 - timeliness_rate,
                                recommendation='Review date validation in source systems'
                            )
                            self.issues.append(issue)
                        
                        timeliness_results.append({
                            'test': 'Origination_Date_Timeliness',
                            'field': orig_col,
                            'valid_records': int(valid_mask.sum()),
                            'total_records': int(total_dates),
                            'timeliness_rate': round(timeliness_rate, 2),
                            'future_dates': int(future_dates),
                            'oldest_date': orig_dates.min().isoformat() if not orig_dates.isna().all() else None,
                            'newest_date': orig_dates.max().isoformat() if not orig_dates.isna().all() else None,
                            'status': status,
                            'severity': severity.value if status != ValidationStatus.PASS.value else None
                        })
                except Exception as e:
                    logger.error(f"Origination date timeliness check failed: {e}")
                    timeliness_results.append({
                        'test': 'Origination_Date_Timeliness',
                        'field': orig_col,
                        'status': ValidationStatus.ERROR.value,
                        'error': str(e)
                    })
        
        return timeliness_results
    
    @_time_operation.__get__(object, type(None))
    def _assess_dimensionality(self) -> List[Dict]:
        """
        Assess data dimensionality and cardinality
        """
        logger.info("Assessing Dimensionality...")
        dimensionality_results = []
        
        for col in self.df.columns:
            unique_values = self.df[col].nunique()
            total_values = self.df[col].notna().sum()
            cardinality_ratio = (unique_values / total_values * 100) if total_values > 0 else 0
            
            # Determine cardinality type
            if cardinality_ratio > 90:
                cardinality_type = "High (Identifier)"
            elif cardinality_ratio > 30:
                cardinality_type = "Medium"
            elif cardinality_ratio > 1:
                cardinality_type = "Low"
            else:
                cardinality_type = "Constant"
            
            # Check for potential issues
            issues = []
            if cardinality_ratio == 100 and col not in ['customer_id', 'account_id']:
                issues.append("Potentially unique field - consider if this should be an identifier")
            elif cardinality_ratio == 0:
                issues.append("Constant field - no variability")
            elif cardinality_ratio < 1 and unique_values > 1:
                issues.append("Very low cardinality - may be categorical")
            
            dimensionality_results.append({
                'field': col,
                'unique_values': unique_values,
                'total_non_null': total_values,
                'cardinality_ratio': round(cardinality_ratio, 2),
                'cardinality_type': cardinality_type,
                'data_type': str(self.df[col].dtype),
                'sample_values': self.df[col].dropna().unique().tolist()[:5] if total_values > 0 else [],
                'potential_issues': issues
            })
        
        return dimensionality_results
    
    @_time_operation.__get__(object, type(None))
    def _assess_distributions(self) -> Dict:
        """
        Assess statistical distributions of key numeric fields
        """
        logger.info("Assessing Distributions...")
        distribution_results = {}
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            try:
                series = pd.to_numeric(self.df[col], errors='coerce').dropna()
                if len(series) > 0:
                    stats_dict = {
                        'count': int(len(series)),
                        'mean': float(series.mean()),
                        'std': float(series.std()),
                        'min': float(series.min()),
                        '25%': float(series.quantile(0.25)),
                        'median': float(series.median()),
                        '75%': float(series.quantile(0.75)),
                        'max': float(series.max()),
                        'skewness': float(series.skew()),
                        'kurtosis': float(series.kurtosis()),
                        'zeros_percentage': float((series == 0).sum() / len(series) * 100),
                        'negative_percentage': float((series < 0).sum() / len(series) * 100),
                        'outliers_iqr': self._detect_outliers_iqr(series)
                    }
                    
                    # Check for distribution issues
                    issues = []
                    if abs(stats_dict['skewness']) > 2:
                        issues.append(f"High skewness ({stats_dict['skewness']:.2f})")
                    if abs(stats_dict['kurtosis']) > 10:
                        issues.append(f"High kurtosis ({stats_dict['kurtosis']:.2f})")
                    if stats_dict['zeros_percentage'] > 50:
                        issues.append(f"High zero percentage ({stats_dict['zeros_percentage']:.1f}%)")
                    
                    stats_dict['distribution_issues'] = issues
                    distribution_results[col] = stats_dict
            except Exception as e:
                logger.error(f"Failed to calculate distribution for {col}: {e}")
                distribution_results[col] = {'error': str(e)}
        
        return distribution_results
    
    @_time_operation.__get__(object, type(None))
    def _cross_field_validation(self) -> List[Dict]:
        """
        Perform cross-field validation
        """
        logger.info("Performing Cross-Field Validation...")
        cross_validation_results = []
        
        # 1. PD should correlate with stage (higher PD in stage 2/3)
        if 'stage' in self.field_mapping and 'pd_value' in self.field_mapping:
            stage_col = self.field_mapping['stage']
            pd_col = self.field_mapping['pd_value']
            
            if stage_col in self.df.columns and pd_col in self.df.columns:
                try:
                    # Convert stage to numeric
                    stage_series = self.df[stage_col].astype(str).str.extract(r'(\d)')[0].astype(float)
                    pd_series = pd.to_numeric(self.df[pd_col], errors='coerce')
                    
                    # Calculate average PD by stage
                    valid_mask = stage_series.notna() & pd_series.notna()
                    if valid_mask.sum() > 0:
                        avg_pd_by_stage = pd.DataFrame({
                            'stage': stage_series[valid_mask],
                            'pd': pd_series[valid_mask]
                        }).groupby('stage')['pd'].agg(['mean', 'count']).round(4)
                        
                        # Check if PD increases with stage
                        stage_order = sorted(avg_pd_by_stage.index.unique())
                        if len(stage_order) > 1:
                            pd_trend = all(avg_pd_by_stage.loc[stage_order[i], 'mean'] <= 
                                          avg_pd_by_stage.loc[stage_order[i+1], 'mean'] 
                                          for i in range(len(stage_order)-1))
                            
                            if not pd_trend:
                                issue = DataQualityIssue(
                                    type='Cross-Validation',
                                    field=f'{stage_col}, {pd_col}',
                                    description='PD does not consistently increase with stage',
                                    severity=SeverityLevel.MEDIUM,
                                    records_affected=0,
                                    percentage_affected=0,
                                    recommendation='Review PD model calibration by stage'
                                )
                                self.issues.append(issue)
                        
                        cross_validation_results.append({
                            'test': 'PD_Stage_Correlation',
                            'fields': f'{stage_col}, {pd_col}',
                            'average_pd_by_stage': avg_pd_by_stage.to_dict(),
                            'pd_trend_consistent': pd_trend if 'pd_trend' in locals() else 'N/A',
                            'status': ValidationStatus.PASS.value if pd_trend else ValidationStatus.WARNING.value
                        })
                except Exception as e:
                    logger.error(f"PD-Stage correlation check failed: {e}")
        
        # 2. EAD should not exceed original balance significantly
        ead_col = self.field_mapping.get('ead_value')
        bal_col = self.field_mapping.get('balance')
        
        if ead_col and bal_col and ead_col in self.df.columns and bal_col in self.df.columns:
            try:
                ead_series = pd.to_numeric(self.df[ead_col], errors='coerce')
                bal_series = pd.to_numeric(self.df[bal_col], errors='coerce')
                
                valid_mask = ead_series.notna() & bal_series.notna()
                if valid_mask.sum() > 0:
                    ead_to_balance_ratio = (ead_series[valid_mask] / bal_series[valid_mask])
                    
                    # Calculate statistics on the ratio
                    ratio_stats = {
                        'mean': float(ead_to_balance_ratio.mean()),
                        'median': float(ead_to_balance_ratio.median()),
                        'std': float(ead_to_balance_ratio.std()),
                        'min': float(ead_to_balance_ratio.min()),
                        'max': float(ead_to_balance_ratio.max()),
                        'p95': float(ead_to_balance_ratio.quantile(0.95)),
                        'above_1.5': float((ead_to_balance_ratio > 1.5).sum() / len(ead_to_balance_ratio) * 100)
                    }
                    
                    if ratio_stats['above_1.5'] > 5:
                        issue = DataQualityIssue(
                            type='Cross-Validation',
                            field=f'{ead_col}, {bal_col}',
                            description=f'EAD exceeds balance by >50% for {ratio_stats["above_1.5"]:.1f}% of records',
                            severity=SeverityLevel.MEDIUM,
                            records_affected=int((ead_to_balance_ratio > 1.5).sum()),
                            percentage_affected=ratio_stats['above_1.5'],
                            recommendation='Review EAD calculation methodology'
                        )
                        self.issues.append(issue)
                    
                    cross_validation_results.append({
                        'test': 'EAD_Balance_Ratio',
                        'fields': f'{ead_col}, {bal_col}',
                        'ratio_statistics': ratio_stats,
                        'status': ValidationStatus.PASS.value if ratio_stats['above_1.5'] <= 5 else ValidationStatus.WARNING.value
                    })
            except Exception as e:
                logger.error(f"EAD-Balance ratio check failed: {e}")
        
        return cross_validation_results
    
    def _calculate_overall_score(self) -> Dict:
        """
        Calculate comprehensive data quality score
        """
        category_scores = {}
        total_weight = 0
        weighted_sum = 0
        
        # Define category weights
        weights = {
            'completeness': 0.25,
            'accuracy': 0.25,
            'consistency': 0.15,
            'validity': 0.15,
            'uniqueness': 0.10,
            'timeliness': 0.10
        }
        
        for category, weight in weights.items():
            if category in self.results:
                category_data = self.results[category]
                
                # Calculate average score for this category
                scores = []
                for item in category_data:
                    if 'status' in item and item['status'] == ValidationStatus.PASS.value:
                        scores.append(100)
                    elif 'status' in item and item['status'] == ValidationStatus.WARNING.value:
                        scores.append(70)
                    elif 'status' in item and item['status'] == ValidationStatus.FAIL.value:
                        scores.append(30)
                
                if scores:
                    category_score = np.mean(scores)
                    category_scores[category] = round(category_score, 1)
                    weighted_sum += category_score * weight
                    total_weight += weight
        
        # Calculate overall weighted score
        if total_weight > 0:
            overall_score = weighted_sum / total_weight
        else:
            overall_score = 0
        
        # Apply penalties for critical issues
        critical_issues_count = len([i for i in self.issues if i.severity == SeverityLevel.CRITICAL])
        penalty = min(critical_issues_count * 3, 30)  # Max 30% penalty
        
        final_score = max(overall_score - penalty, 0)
        
        return {
            'overall_score': round(final_score, 1),
            'category_scores': category_scores,
            'critical_issues_count': critical_issues_count,
            'penalty_applied': penalty,
            'grade': self._score_to_grade(final_score)
        }
    
    def _calculate_risk_score(self) -> Dict:
        """
        Calculate risk-based quality score for IFRS 9
        """
        risk_factors = {
            'pd_invalid': 0,
            'lgd_invalid': 0,
            'ead_invalid': 0,
            'stage_invalid': 0,
            'balance_invalid': 0,
            'dates_invalid': 0,
            'duplicates': 0
        }
        
        # Calculate risk factors based on issues found
        for issue in self.issues:
            if 'pd' in issue.field.lower() or 'probability' in issue.field.lower():
                risk_factors['pd_invalid'] += issue.percentage_affected * 0.5
            elif 'lgd' in issue.field.lower():
                risk_factors['lgd_invalid'] += issue.percentage_affected * 0.5
            elif 'ead' in issue.field.lower() or 'exposure' in issue.field.lower():
                risk_factors['ead_invalid'] += issue.percentage_affected * 0.3
            elif 'stage' in issue.field.lower() or 'class' in issue.field.lower():
                risk_factors['stage_invalid'] += issue.percentage_affected * 0.4
            elif 'balance' in issue.field.lower():
                risk_factors['balance_invalid'] += issue.percentage_affected * 0.2
            elif 'date' in issue.field.lower():
                risk_factors['dates_invalid'] += issue.percentage_affected * 0.1
            elif 'duplicate' in issue.description.lower():
                risk_factors['duplicates'] += issue.percentage_affected * 0.2
        
        # Calculate total risk score (0-100, higher is riskier)
        total_risk = sum(risk_factors.values())
        normalized_risk = min(total_risk, 100)
        
        return {
            'risk_score': round(normalized_risk, 1),
            'risk_factors': risk_factors,
            'risk_level': self._risk_to_level(normalized_risk)
        }
    
    def _get_summary_statistics(self) -> Dict:
        """
        Get summary statistics for the dataset
        """
        stats_dict = {
            'total_records': len(self.df),
            'total_columns': len(self.df.columns),
            'total_null_values': self.df.isna().sum().sum(),
            'null_percentage': (self.df.isna().sum().sum() / (len(self.df) * len(self.df.columns))) * 100,
            'duplicate_records': len(self.df) - len(self.df.drop_duplicates()),
            'data_types': dict(self.df.dtypes.value_counts()),
            'key_fields_mapped': len([k for k, v in self.field_mapping.items() if v]),
            'assessment_timestamp': datetime.now().isoformat()
        }
        
        # Add IFRS 9 specific statistics if fields exist
        ifrs9_stats = {}
        
        if 'stage' in self.field_mapping and self.field_mapping['stage'] in self.df.columns:
            stage_col = self.field_mapping['stage']
            stage_counts = self.df[stage_col].value_counts().to_dict()
            ifrs9_stats['stage_distribution'] = stage_counts
        
        if 'pd_value' in self.field_mapping and self.field_mapping['pd_value'] in self.df.columns:
            pd_col = self.field_mapping['pd_value']
            pd_series = pd.to_numeric(self.df[pd_col], errors='coerce')
            ifrs9_stats['pd_statistics'] = {
                'mean': float(pd_series.mean()),
                'median': float(pd_series.median()),
                'std': float(pd_series.std()),
                'min': float(pd_series.min()),
                'max': float(pd_series.max())
            }
        
        stats_dict['ifrs9_statistics'] = ifrs9_stats
        
        return stats_dict
    
    def _categorize_issues(self) -> Dict:
        """
        Categorize issues by severity and type
        """
        issues_by_severity = {}
        issues_by_type = {}
        
        for issue in self.issues:
            # By severity
            severity = issue.severity.value
            if severity not in issues_by_severity:
                issues_by_severity[severity] = []
            issues_by_severity[severity].append({
                'type': issue.type,
                'field': issue.field,
                'description': issue.description,
                'records_affected': issue.records_affected,
                'percentage_affected': issue.percentage_affected
            })
            
            # By type
            issue_type = issue.type
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append({
                'field': issue.field,
                'description': issue.description,
                'severity': severity,
                'records_affected': issue.records_affected
            })
        
        return {
            'by_severity': issues_by_severity,
            'by_type': issues_by_type,
            'total_issues': len(self.issues),
            'critical_issues': len([i for i in self.issues if i.severity == SeverityLevel.CRITICAL])
        }
    
    def _generate_recommendations(self) -> List[Dict]:
        """
        Generate actionable recommendations based on findings
        """
        recommendations = []
        
        # Analyze issues and generate recommendations
        critical_issues = [i for i in self.issues if i.severity == SeverityLevel.CRITICAL]
        high_issues = [i for i in self.issues if i.severity == SeverityLevel.HIGH]
        
        # Group issues by field for targeted recommendations
        field_issues = {}
        for issue in self.issues:
            if issue.field not in field_issues:
                field_issues[issue.field] = []
            field_issues[issue.field].append(issue)
        
        # Generate field-specific recommendations
        for field, issues in field_issues.items():
            if len(issues) > 0:
                critical_count = len([i for i in issues if i.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]])
                if critical_count > 0:
                    recommendations.append({
                        'field': field,
                        'priority': 'High' if critical_count > 0 else 'Medium',
                        'issues_count': len(issues),
                        'recommendation': f'Review and validate {field} data quality. Consider implementing automated validation rules.',
                        'affected_percentage': max(i.percentage_affected for i in issues)
                    })
        
        # General recommendations based on overall score
        overall_score = self.results['overall_score']['overall_score']
        
        if overall_score < 70:
            recommendations.append({
                'field': 'Overall',
                'priority': 'Critical',
                'recommendation': 'Data quality is below acceptable levels. Conduct thorough data cleansing before IFRS 9 calculations.',
                'action': 'Immediate data remediation required'
            })
        elif overall_score < 85:
            recommendations.append({
                'field': 'Overall',
                'priority': 'High',
                'recommendation': 'Moderate data quality issues found. Address critical issues before month-end reporting.',
                'action': 'Prioritize critical issue resolution'
            })
        
        # Add IFRS 9 specific recommendations
        if 'stage' in self.field_mapping:
            recommendations.append({
                'field': 'Stage Classification',
                'priority': 'Medium',
                'recommendation': 'Regularly validate stage migration triggers and criteria compliance.',
                'action': 'Monthly validation process'
            })
        
        return recommendations
    
    def _print_comprehensive_summary(self):
        """
        Print comprehensive assessment summary
        """
        print(f"\n{'='*80}")
        print(f"ASSESSMENT SUMMARY - {self.bank_name}")
        print(f"{'='*80}")
        
        # Overall scores
        overall = self.results['overall_score']
        risk = self.results['risk_score']
        
        print(f"\n📊 OVERALL DATA QUALITY SCORE: {overall['overall_score']}/100 ({overall['grade']})")
        print(f"⚠️  RISK SCORE: {risk['risk_score']}/100 ({risk['risk_level']})")
        print(f"🚨 CRITICAL ISSUES: {overall['critical_issues_count']}")
        print(f"⏱️  ASSESSMENT DURATION: {self.results['performance']['total_duration_seconds']:.2f} seconds")
        
        # Category scores
        print(f"\n{'='*50}")
        print(f"CATEGORY SCORES")
        print(f"{'='*50}")
        for category, score in overall['category_scores'].items():
            print(f"{category.title():15s}: {score:6.1f}/100")
        
        # Issue summary
        issues_summary = self.results['issues_by_severity']
        print(f"\n{'='*50}")
        print(f"ISSUES SUMMARY")
        print(f"{'='*50}")
        for severity, issues in issues_summary['by_severity'].items():
            print(f"{severity:15s}: {len(issues):3d} issues")
        
        # Key findings
        print(f"\n{'='*50}")
        print(f"KEY FINDINGS")
        print(f"{'='*50}")
        
        stats = self.results['summary_statistics']
        print(f"Total Records: {stats['total_records']:,}")
        print(f"Null Values: {stats['null_percentage']:.1f}%")
        print(f"Duplicate Records: {stats['duplicate_records']:,}")
        
        # IFRS 9 specific findings
        ifrs9_stats = stats.get('ifrs9_statistics', {})
        if 'stage_distribution' in ifrs9_stats:
            print(f"\nIFRS 9 Stage Distribution:")
            for stage, count in ifrs9_stats['stage_distribution'].items():
                percentage = (count / stats['total_records']) * 100
                print(f"  Stage {stage}: {count:,} ({percentage:.1f}%)")
        
        # Top recommendations
        recommendations = self.results['recommendations']
        if recommendations:
            print(f"\n{'='*50}")
            print(f"TOP RECOMMENDATIONS")
            print(f"{'='*50}")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"{i}. [{rec['priority']}] {rec['recommendation']}")
        
        print(f"\n{'='*80}")
        print(f"Assessment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
    
    # ===================================================================
    # HELPER METHODS
    # ===================================================================
    
    def _parse_date_column(self, column_name: str) -> Optional[pd.Series]:
        """
        Robust date parsing with multiple format attempts
        """
        if column_name not in self.df.columns:
            return None
        
        series = self.df[column_name]
        
        # Try multiple date formats
        for fmt in self.config['date_formats']:
            try:
                parsed = pd.to_datetime(series, format=fmt, errors='coerce')
                if parsed.notna().sum() > 0:
                    return parsed
            except:
                continue
        
        # Final attempt with mixed format
        return pd.to_datetime(series, errors='coerce')
    
    def _detect_outliers_iqr(self, series: pd.Series) -> Dict:
        """
        Detect outliers using IQR method
        """
        if len(series) < 4:
            return {'outlier_count': 0, 'outlier_percentage': 0}
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        return {
            'outlier_count': int(len(outliers)),
            'outlier_percentage': float((len(outliers) / len(series)) * 100),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound)
        }
    
    def _score_to_grade(self, score: float) -> str:
        """
        Convert score to letter grade
        """
        if score >= 90:
            return "Excellent (A)"
        elif score >= 80:
            return "Good (B)"
        elif score >= 70:
            return "Fair (C)"
        elif score >= 60:
            return "Poor (D)"
        else:
            return "Critical (F)"
    
    def _risk_to_level(self, risk_score: float) -> str:
        """
        Convert risk score to risk level
        """
        if risk_score >= 70:
            return "High Risk"
        elif risk_score >= 40:
            return "Medium Risk"
        elif risk_score >= 20:
            return "Low Risk"
        else:
            return "Minimal Risk"
    
    def _fuzzy_match(self, target: str, candidates: List[str]) -> Optional[str]:
        """
        Simple fuzzy matching (placeholder - can be enhanced with libraries like fuzzywuzzy)
        """
        target_lower = target.lower()
        best_match = None
        best_score = 0
        
        for candidate in candidates:
            candidate_lower = candidate.lower()
            
            # Simple scoring based on common substrings
            score = 0
            if target_lower in candidate_lower:
                score += 3
            if candidate_lower in target_lower:
                score += 2
            
            # Check for common words
            target_words = set(re.findall(r'\b\w+\b', target_lower))
            candidate_words = set(re.findall(r'\b\w+\b', candidate_lower))
            common_words = target_words.intersection(candidate_words)
            score += len(common_words)
            
            if score > best_score and score > 1:
                best_score = score
                best_match = candidate
        
        return best_match
    
    # ===================================================================
    # EXPORT AND VISUALIZATION METHODS
    # ===================================================================
    
    def export_results(self, format: str = 'json', output_path: Optional[str] = None) -> str:
        """
        Export assessment results to various formats
        
        Args:
            format: Export format ('json', 'html', 'excel', 'csv')
            output_path: Path to save the file
        
        Returns:
            Exported content or file path
        """
        if not self.results:
            raise ValueError("No assessment results available. Run assessment first.")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_filename = f"ifrs9_dq_assessment_{self.bank_name}_{timestamp}"
        
        if format.lower() == 'json':
            export_data = json.dumps(self.results, indent=2, default=str)
            filename = f"{default_filename}.json"
            
        elif format.lower() == 'html':
            export_data = self._generate_html_report()
            filename = f"{default_filename}.html"
            
        elif format.lower() == 'excel':
            filename = f"{default_filename}.xlsx"
            export_data = self._export_to_excel(filename)
            
        elif format.lower() == 'csv':
            # Export summary to CSV
            summary_df = self._create_summary_dataframe()
            export_data = summary_df.to_csv(index=False)
            filename = f"{default_filename}_summary.csv"
            
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Save to file if output_path is provided
        if output_path:
            full_path = os.path.join(output_path, filename)
            if format.lower() == 'excel':
                # Already saved in _export_to_excel
                pass
            else:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(export_data)
            return full_path
        
        return export_data
    
    def _create_summary_dataframe(self) -> pd.DataFrame:
        """Create summary DataFrame for CSV export"""
        summary_data = []
        
        # Add overall metrics
        summary_data.append({
            'Category': 'Overall',
            'Metric': 'Data Quality Score',
            'Value': self.results['overall_score']['overall_score'],
            'Grade': self.results['overall_score']['grade']
        })
        
        summary_data.append({
            'Category': 'Overall',
            'Metric': 'Risk Score',
            'Value': self.results['risk_score']['risk_score'],
            'Grade': self.results['risk_score']['risk_level']
        })
        
        # Add category scores
        for category, score in self.results['overall_score']['category_scores'].items():
            summary_data.append({
                'Category': 'Category Scores',
                'Metric': category.title(),
                'Value': score,
                'Grade': ''
            })
        
        # Add issue counts
        issues_summary = self.results['issues_by_severity']
        for severity, issues in issues_summary['by_severity'].items():
            summary_data.append({
                'Category': 'Issues',
                'Metric': f'{severity} Issues',
                'Value': len(issues),
                'Grade': ''
            })
        
        return pd.DataFrame(summary_data)
    
    def _export_to_excel(self, filename: str) -> str:
        """Export detailed results to Excel"""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = self._create_summary_dataframe()
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Issues sheet
            issues_data = []
            for issue in self.issues:
                issues_data.append({
                    'Type': issue.type,
                    'Field': issue.field,
                    'Description': issue.description,
                    'Severity': issue.severity.value,
                    'Records Affected': issue.records_affected,
                    'Percentage Affected': f"{issue.percentage_affected:.1f}%",
                    'Recommendation': issue.recommendation
                })
            
            if issues_data:
                issues_df = pd.DataFrame(issues_data)
                issues_df.to_excel(writer, sheet_name='Issues', index=False)
            
            # Field mapping sheet
            mapping_data = []
            for std_name, actual_name in self.field_mapping.items():
                mapping_data.append({
                    'Standard Name': std_name,
                    'Actual Column': actual_name,
                    'Status': 'Mapped' if actual_name else 'Not Found'
                })
            
            mapping_df = pd.DataFrame(mapping_data)
            mapping_df.to_excel(writer, sheet_name='Field Mapping', index=False)
        
        return filename
    
    def _generate_html_report(self) -> str:
        """Generate HTML report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>IFRS 9 Data Quality Assessment - {bank_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                .score {{ font-size: 36px; font-weight: bold; color: #3498db; }}
                .metric {{ margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-left: 4px solid #3498db; }}
                .critical {{ color: #e74c3c; font-weight: bold; }}
                .warning {{ color: #f39c12; }}
                .good {{ color: #27ae60; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #34495e; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>IFRS 9 Data Quality Assessment</h1>
                <h2>{bank_name} - {assessment_date}</h2>
            </div>
            
            <div class="metric">
                <h3>Overall Data Quality Score: <span class="score">{overall_score}/100</span></h3>
                <p>Grade: {grade} | Risk Score: {risk_score}/100 ({risk_level})</p>
            </div>
            
            <h3>Category Scores</h3>
            <table>
                <tr><th>Category</th><th>Score</th></tr>
                {category_rows}
            </table>
            
            <h3>Key Statistics</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                {stat_rows}
            </table>
            
            <h3>Issues Summary</h3>
            <table>
                <tr><th>Severity</th><th>Count</th></tr>
                {issue_rows}
            </table>
            
            <h3>Top Recommendations</h3>
            <ul>
                {recommendation_items}
            </ul>
            
            <p><em>Report generated: {generated_time}</em></p>
        </body>
        </html>
        """
        
        # Prepare data for template
        overall = self.results['overall_score']
        risk = self.results['risk_score']
        stats = self.results['summary_statistics']
        
        # Category rows
        category_rows = ""
        for category, score in overall['category_scores'].items():
            category_rows += f"<tr><td>{category.title()}</td><td>{score}/100</td></tr>\n"
        
        # Stat rows
        stat_rows = f"""
        <tr><td>Total Records</td><td>{stats['total_records']:,}</td></tr>
        <tr><td>Total Columns</td><td>{stats['total_columns']}</td></tr>
        <tr><td>Null Percentage</td><td>{stats['null_percentage']:.1f}%</td></tr>
        <tr><td>Duplicate Records</td><td>{stats['duplicate_records']:,}</td></tr>
        """
        
        # Issue rows
        issue_rows = ""
        issues_summary = self.results['issues_by_severity']['by_severity']
        for severity, issues in issues_summary.items():
            issue_rows += f"<tr><td>{severity}</td><td>{len(issues)}</td></tr>\n"
        
        # Recommendation items
        recommendation_items = ""
        recommendations = self.results['recommendations'][:5]
        for rec in recommendations:
            recommendation_items += f"<li>[{rec['priority']}] {rec['recommendation']}</li>\n"
        
        # Fill template
        html_content = html_template.format(
            bank_name=self.bank_name,
            assessment_date=self.assessment_date.strftime('%Y-%m-%d'),
            overall_score=overall['overall_score'],
            grade=overall['grade'],
            risk_score=risk['risk_score'],
            risk_level=risk['risk_level'],
            category_rows=category_rows,
            stat_rows=stat_rows,
            issue_rows=issue_rows,
            recommendation_items=recommendation_items,
            generated_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        return html_content
    
    def visualize_results(self, output_path: Optional[str] = None):
        """
        Create visualizations of assessment results
        
        Args:
            output_path: Path to save visualizations
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.style.use('seaborn-v0_8-darkgrid')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 10))
            
            # 1. Overall score gauge
            ax1 = plt.subplot(2, 3, 1)
            self._create_score_gauge(ax1)
            
            # 2. Category scores bar chart
            ax2 = plt.subplot(2, 3, 2)
            self._create_category_barchart(ax2)
            
            # 3. Issues by severity pie chart
            ax3 = plt.subplot(2, 3, 3)
            self._create_issues_piechart(ax3)
            
            # 4. Null percentage heatmap
            ax4 = plt.subplot(2, 3, 4)
            self._create_null_heatmap(ax4)
            
            # 5. PD distribution (if available)
            ax5 = plt.subplot(2, 3, 5)
            self._create_pd_distribution(ax5)
            
            # 6. Stage distribution (if available)
            ax6 = plt.subplot(2, 3, 6)
            self._create_stage_distribution(ax6)
            
            plt.suptitle(f'IFRS 9 Data Quality Assessment - {self.bank_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to: {output_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Visualization requires matplotlib and seaborn. Install with: pip install matplotlib seaborn")
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    def _create_score_gauge(self, ax):
        """Create gauge chart for overall score"""
        score = self.results['overall_score']['overall_score']
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        ax.plot(theta, r, color='black', linewidth=2)
        
        # Fill based on score
        score_angle = (score / 100) * np.pi
        theta_fill = np.linspace(0, score_angle, 100)
        r_fill = np.ones_like(theta_fill)
        
        # Color based on score
        if score >= 80:
            color = 'green'
        elif score >= 60:
            color = 'orange'
        else:
            color = 'red'
        
        ax.fill_between(theta_fill, 0, r_fill, color=color, alpha=0.6)
        
        # Add score text
        ax.text(score_angle/2, 0.5, f'{score:.0f}', ha='center', va='center', 
                fontsize=24, fontweight='bold')
        
        ax.set_title('Overall Quality Score', fontsize=12, fontweight='bold')
        ax.axis('off')
        ax.set_xlim(0, np.pi)
        ax.set_ylim(0, 1.2)
    
    def _create_category_barchart(self, ax):
        """Create bar chart for category scores"""
        category_scores = self.results['overall_score']['category_scores']
        
        categories = list(category_scores.keys())
        scores = list(category_scores.values())
        
        colors = ['green' if s >= 80 else 'orange' if s >= 60 else 'red' for s in scores]
        
        bars = ax.bar(categories, scores, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{score:.0f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylim(0, 105)
        ax.set_ylabel('Score')
        ax.set_title('Category Scores', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _create_issues_piechart(self, ax):
        """Create pie chart for issues by severity"""
        issues_summary = self.results['issues_by_severity']['by_severity']
        
        if not issues_summary:
            ax.text(0.5, 0.5, 'No Issues Found', ha='center', va='center', fontsize=12)
            ax.set_title('Issues by Severity', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        labels = list(issues_summary.keys())
        sizes = [len(issues) for issues in issues_summary.values()]
        
        # Colors for severities
        severity_colors = {
            'Critical': 'red',
            'High': 'orange',
            'Medium': 'yellow',
            'Low': 'lightblue',
            'Info': 'lightgray'
        }
        
        colors = [severity_colors.get(label, 'gray') for label in labels]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                         autopct='%1.1f%%', startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Issues by Severity', fontsize=12, fontweight='bold')
        ax.axis('equal')
    
    def _create_null_heatmap(self, ax):
        """Create heatmap of null values by column"""
        null_percentages = []
        columns = []
        
        for col in self.df.columns:
            null_pct = (self.df[col].isna().sum() / len(self.df)) * 100
            if null_pct > 0:  # Only show columns with nulls
                null_percentages.append(null_pct)
                columns.append(col[:20] + '...' if len(col) > 20 else col)
        
        if not null_percentages:
            ax.text(0.5, 0.5, 'No Null Values', ha='center', va='center', fontsize=12)
            ax.set_title('Null Values Heatmap', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        # Create heatmap
        colors = plt.cm.Reds(np.array(null_percentages) / 100)
        y_pos = np.arange(len(columns))
        
        ax.barh(y_pos, null_percentages, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(columns)
        ax.set_xlabel('Null Percentage (%)')
        ax.set_title('Null Values by Column', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _create_pd_distribution(self, ax):
        """Create PD distribution histogram"""
        if 'pd_value' in self.field_mapping:
            pd_col = self.field_mapping['pd_value']
            if pd_col in self.df.columns:
                pd_series = pd.to_numeric(self.df[pd_col], errors='coerce').dropna()
                
                if len(pd_series) > 0:
                    ax.hist(pd_series, bins=50, alpha=0.7, color='blue', edgecolor='black')
                    ax.set_xlabel('PD Value')
                    ax.set_ylabel('Frequency')
                    ax.set_title('PD Distribution', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    return
        
        ax.text(0.5, 0.5, 'PD Data\nNot Available', ha='center', va='center', fontsize=12)
        ax.set_title('PD Distribution', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    def _create_stage_distribution(self, ax):
        """Create stage distribution bar chart"""
        if 'stage' in self.field_mapping:
            stage_col = self.field_mapping['stage']
            if stage_col in self.df.columns:
                stage_counts = self.df[stage_col].value_counts()
                
                if len(stage_counts) > 0:
                    stage_counts.plot(kind='bar', ax=ax, color=['green', 'orange', 'red'], alpha=0.7)
                    ax.set_xlabel('Stage')
                    ax.set_ylabel('Count')
                    ax.set_title('Stage Distribution', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    return
        
        ax.text(0.5, 0.5, 'Stage Data\nNot Available', ha='center', va='center', fontsize=12)
        ax.set_title('Stage Distribution', fontsize=12, fontweight='bold')
        ax.axis('off')


# ===================================================================
# CONVENIENCE FUNCTIONS
# ===================================================================

def load_and_assess_data(file_path: str, bank_name: str = None, 
                        config_file: Optional[str] = None) -> IFRS9DataQualityFramework:
    """
    Convenience function to load data and run assessment
    
    Args:
        file_path: Path to data file (CSV, Excel, XLSB)
        bank_name: Name of the bank
        config_file: Path to configuration file
    
    Returns:
        Initialized and assessed framework
    """
    # Extract bank name from filename if not provided
    if bank_name is None:
        bank_name = Path(file_path).stem.replace('_', ' ').title()
    
    # Load data based on file extension
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(file_path)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    elif file_ext == '.xlsb':
        # Use pyxlsb for binary Excel files
        with pyxlsb.open_workbook(file_path) as wb:
            with wb.get_sheet(1) as sheet:
                data = list(sheet.rows())
        # Convert to DataFrame
        headers = [cell.v for cell in data[0]]
        rows = [[cell.v for cell in row] for row in data[1:]]
        df = pd.DataFrame(rows, columns=headers)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Initialize and run assessment
    framework = IFRS9DataQualityFramework(df, bank_name=bank_name, config_file=config_file)
    framework.run_comprehensive_assessment()
    
    return framework


def compare_assessments(frameworks: List[IFRS9DataQualityFramework]) -> pd.DataFrame:
    """
    Compare multiple assessments
    
    Args:
        frameworks: List of IFRS9DataQualityFramework instances
    
    Returns:
        Comparison DataFrame
    """
    comparison_data = []
    
    for framework in frameworks:
        overall = framework.results['overall_score']
        risk = framework.results['risk_score']
        stats = framework.results['summary_statistics']
        
        comparison_data.append({
            'Bank': framework.bank_name,
            'Data Quality Score': overall['overall_score'],
            'Grade': overall['grade'],
            'Risk Score': risk['risk_score'],
            'Risk Level': risk['risk_level'],
            'Total Records': stats['total_records'],
            'Total Columns': stats['total_columns'],
            'Null Percentage': f"{stats['null_percentage']:.1f}%",
            'Critical Issues': overall['critical_issues_count'],
            'Total Issues': framework.results['issues_by_severity']['total_issues']
        })
    
    return pd.DataFrame(comparison_data)


# ===================================================================
# MAIN EXECUTION
# ===================================================================

if __name__ == "__main__":
    """
    Example usage of the enhanced IFRS 9 Data Quality Framework
    """
    print("IFRS 9 Enhanced Data Quality Framework")
    print("=" * 50)
    
    # Example 1: Create sample data for demonstration
    print("\nExample 1: Creating sample data...")
    sample_data = {
        'CUSTID': ['C001', 'C002', 'C003', 'C004', 'C005'],
        'AA.ID': ['ACC001', 'ACC002', 'ACC003', 'ACC004', 'ACC005'],
        'SUSPD.AMT': [0.01, 0.05, 0.10, 0.15, 0.20],
        'lgd': [0.45, 0.50, 0.55, 0.60, 0.65],
        'ead': [10000, 15000, 20000, 25000, 30000],
        'CLASS': [1, 1, 2, 2, 3],
        'C.LOAN.BAL': [9500, 14000, 19000, 24000, 29000],
        'START.DATE': ['2022-01-15', '2022-03-20', '2021-11-10', '2020-08-05', '2019-05-30'],
        'MATURITY.DATE': ['2027-01-15', '2027-03-20', '2026-11-10', '2025-08-05', '2024-05-30'],
        'INT.RATE': [0.08, 0.09, 0.10, 0.11, 0.12],
        'SECTOR': ['Retail', 'Commercial', 'Retail', 'Commercial', 'Retail'],
        'dpd': [0, 15, 45, 90, 120]
    }
    
    df_sample = pd.DataFrame(sample_data)
    
    # Example 2: Run assessment on sample data
    print("\nExample 2: Running assessment...")
    framework = IFRS9DataQualityFramework(df_sample, bank_name="Sample Bank")
    results = framework.run_comprehensive_assessment()
    
    # Example 3: Export results
    print("\nExample 3: Exporting results...")
    json_output = framework.export_results(format='json')
    print("JSON export complete (first 500 chars):")
    print(json_output[:500] + "...")
    
    # Example 4: Get recommendations
    print("\nExample 4: Top Recommendations:")
    for i, rec in enumerate(framework.results['recommendations'][:3], 1):
        print(f"{i}. [{rec['priority']}] {rec['recommendation']}")
    
    print("\n" + "="*50)
    print("Assessment complete!")