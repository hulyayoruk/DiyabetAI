# ml/feature_config.py

# XGBoost ve LSTM için kullanılan feature sırası
# build_lstm_window'da tam olarak BU sıraya göre array üreteceğiz.
XGB_FEATURE_ORDER = [
    "glucose",          # 0
    "carbs_win_1",      # 1
    "carbs_win_2",      # 2
    "carbs_win_3",      # 3
    "bolus",            # 4 (şimdilik 0)
    "bolus_corr",       # 5 (şimdilik 0)
    "basal",            # 6 (şimdilik 0)
    "ex_minutes",       # 7
    "ex_intensity",     # 8
    "steps",            # 9
    "is_sleep",         # 10
    "time_sin",         # 11
    "time_cos",         # 12
]
