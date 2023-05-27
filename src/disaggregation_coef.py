def disaggregation_coef():
    """
    Function to define time intervals and their associated coefficients for disaggregation.
    Returns:
        tuple: Returns a tuple containing two dictionaries. 
               The first dictionary maps the time intervals to their duration in hours. 
               The second dictionary maps the time intervals to their respective coefficients.
    """

    time_intervals = {
        "24h": 24,
        "12h": 12,
        "10h": 10,
        "8h": 8,
        "6h": 6,
        "4h": 4,
        "2h": 2,
        "1h": 1,
        "30min": 0.5,
        "25min": 25/60,
        "20min": 20/60,
        "15min": 15/60,
        "10min": 10/60,
        "5min": 5/60
    }

    coefficients = {
        "24h": 1.14,
        "12h": 0.85,
        "10h": 0.82,
        "8h": 0.78,
        "6h": 0.72,
        "4h": 0.63,
        "2h": 0.52,
        "1h": 0.42,
        "30min": 0.311,
        "25min": 0.283,
        "20min": 0.252,
        "15min": 0.218,
        "10min": 0.168,
        "5min": 0.106,
    }
    
    return coefficients, time_intervals
