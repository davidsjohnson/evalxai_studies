import numpy as np

def calc_RAIR(y_true, y_model, y_user_base, y_user_advised):
    """
    Calculate the RAIR (Relative AI Reliance) metric.
    
    Schemmer et al (2023) Appropriate Reliance on AI Advice: 
    Conceptualization and the Effect of Explanations

    RAIR is defined...

    Args:
        y_true (list or np.array): True values.
        y_model (list or np.array): Model predicted values.
        y_user_base (list or np.array): User predicted values before advice.
        y_user_advised (list or np.array): User predicted values after advice.
    Returns:
        float: RAIR score.
    """

    y_true = np.array(y_true)
    y_model = np.array(y_model)
    y_user_base = np.array(y_user_base)
    y_user_advised = np.array(y_user_advised)

    df = {
        'true': y_true,
        'model': y_model,
        'user_base': y_user_base,
        'user_advised': y_user_advised
    }

    # CAIR: the case when the human is initially incorrect, 
    # receives correct advice, and relies on that advice
    cair_cases = (
        (df['user_base'] != df['true']) & 
        (df['model'] == df['true']) & 
        (df['user_advised'] == df['model'])
    )

    # CA: is one if the original human decision was wrong and the 
    # AI advice was correct, regardless of the final human decision, 
    # and zero otherwise.
    ca_cases = (
        (df['user_base'] != df['true']) & 
        (df['model'] == df['true'])
    )

    # RAIR: proportion of CAIR cases out of CA cases
    rair = cair_cases.sum() / ca_cases.sum() if ca_cases.sum() > 0 else 0.0

    return rair

def calc_RSR(y_true, y_model, y_user_base, y_user_advised):
    """
    Calculate the RSR (Relative Self-Reliance) metric.
    
    Schemmer et al (2023) Appropriate Reliance on AI Advice: 
    Conceptualization and the Effect of Explanations

    RSR is defined...

    Args:
        y_true (list or np.array): True values.
        y_model (list or np.array): Model predicted values.
        y_user_base (list or np.array): User predicted values before advice.
        y_user_advised (list or np.array): User predicted values after advice.
    Returns:
        float: RSR score.
    """

    y_true = np.array(y_true)
    y_model = np.array(y_model)
    y_user_base = np.array(y_user_base)
    y_user_advised = np.array(y_user_advised)

    df = {
        'true': y_true,
        'model': y_model,
        'user_base': y_user_base,
        'user_advised': y_user_advised
    }

    # CSR (correct self reliance): the case when the human is initially correct, 
    # receives incorrect advice, and relies on that themselves
    csr_cases = (
        (df['user_base'] == df['true']) & 
        (df['model'] != df['true']) & 
        (df['user_advised'] == df['user_base'])
    )

    # IA (incorrect AI advice): the case where the AI is incorrect 
    ia_cases = (
        (df['model'] != df['true'])
    )

    # RSR: proportion of CSR cases out of IA cases
    rsr = csr_cases.sum() / ia_cases.sum() if ia_cases.sum() > 0 else 0.0

    return rsr

def calc_AoR(y_true, y_model, y_user_base, y_user_advised):
    """
    Calculate the AoR (Adherence on Reliance) metric.
    
    Schemmer et al (2023) Appropriate Reliance on AI Advice: 
    Conceptualization and the Effect of Explanations

    AoR is defined...

    Args:
        y_true (list or np.array): True values.
        y_model (list or np.array): Model predicted values.
        y_user_base (list or np.array): User predicted values before advice.
        y_user_advised (list or np.array): User predicted values after advice.
    Returns:
        tuple: rair (float), rsr (float)
    """
    rair = calc_RAIR(y_true, y_model, y_user_base, y_user_advised)
    rsr = calc_RSR(y_true, y_model, y_user_base, y_user_advised)
    return rair, rsr


def test_metrics():
    # Example test cases
    # [CAIR, CSR, IAIR, CAIR, ICR, ICR]
    y_true =         [1, 0, 1, 1, 0, 1, 0, 0]
    y_model =        [1, 0, 0, 1, 1, 0, 0, 1] 
    y_user_base =    [0, 0, 1, 0, 0, 1, 1, 0] 
    y_user_advised = [1, 0, 0, 1, 1, 0, 1, 0]

    # CA Cases: 3 (Indices 0, 3, 6)
    # CAIR Cases: 2 (Indices 0, 3)
    # RAIR = 2 / 3 = 0.6667
    #
    # CSR: 1 (Index 7)
    # IA Cases: 4 (Indices 2, 4, 5, 7)
    # RSR = 1 / 4 = 0.25

    rair = calc_RAIR(y_true, y_model, y_user_base, y_user_advised)
    rsr = calc_RSR(y_true, y_model, y_user_base, y_user_advised)

    assert abs(rair - 0.6667) < 1e-4, f"Expected RAIR ~0.6667, got {rair}"
    assert abs(rsr - 0.25) < 1e-4, f"Expected RSR ~0.25, got {rsr}"

if __name__ == "__main__":
    test_metrics()


# NOTES
# one dataframe per condition (hs / ls)
# pid, bid, y_true, y_pred, y_user_base, y_user_advised