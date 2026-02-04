class Config:
    sensitivity_analyses = {
        # Choice of emissions or carbon intensity first
        'emissions': 0,
        'carbon_intensity': 1,
        # Choice of tests
        'method_test': 1,
        'lowess_fraction_test': 0,
        'noise_distribution_test': 1,
        'aic_bic_comparison': 1,
    }
    