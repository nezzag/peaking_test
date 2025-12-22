class Config:
    sensitivity_analyses = {
        # Choice of emissions or carbon intensity first
        'emissions': True,
        'carbon_intensity': False,
        # Choice of tests
        'method_test': True,
        'lowess_fraction_test': False,
        'noise_distribution_test': False,
        'aic_bic_comparison': False,
    }
    