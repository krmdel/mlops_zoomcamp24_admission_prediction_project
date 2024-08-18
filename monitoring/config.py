# Defining numerical and categorical features
numerical_features = ["age", "albumin_last", "albumin_max", "albumin_median", "albumin_min", "bloodculture,routine_count",  "bloodculture,routine_last",
                      "bloodculture,routine_npos", "cc_abdominalcramping", "cc_abdominaldistention", "cc_abdominalpain", "cc_abdominalpainpregnant",
                      "cc_allergicreaction", "cc_bleeding/bruising", "cc_breastpain", "cc_chestpain", "cc_confusion", "cc_diarrhea",
                      "cc_dizziness", "cc_fall>65", "cc_fever", "cc_hallucinations", "cc_headache", "cc_hypertension", "cc_hypotension",
                      "cc_irregularheartbeat", "cc_nausea", "cc_overdose-accidental", "cc_overdose-intentional", "cc_poisoning", "cc_rapidheartrate",
                      "cc_rectalbleeding", "cc_strokealert", "cc_unresponsive", "cc_urinaryretention", "cktotal_last", "cktotal_max",
                      "cktotal_median", "cktotal_min", "d-dimer_last", "d-dimer_max", "d-dimer_median", "d-dimer_min", "esi", "n_admissions", "n_edvisits", "n_surgeries", "platelets_last", "platelets_max", "platelets_median", "platelets_min",
                      "rbc_last", "rbc_max", "rbc_median", "rbc_min", "triage_vital_dbp", "triage_vital_hr", "triage_vital_o2",
                      "triage_vital_o2_device", "triage_vital_rr", "triage_vital_sbp", "triage_vital_temp", "troponini(poc)_last", "troponini(poc)_max",
                      "troponini(poc)_median", "troponini(poc)_min", "troponint_last", "troponint_max", "troponint_median", "troponint_min",
                      "urineculture,routine_count", "urineculture,routine_last", "urineculture,routine_npos", "viralinfect", "wbc_last",
                      "wbc_max", "wbc_median", "wbc_min"]

categorical_features = ['arrivalmode', 'gender', 'previousdispo']

# Defining target
target = 'disposition' # admit or discharge converted into 1 or 0
