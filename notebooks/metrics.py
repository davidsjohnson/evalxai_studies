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
        df['user_base'] != df['true'] & 
        (df['model'] == df['true']) & 
        (df['user_advised'] == df['model'])
    )

    # CA: is one if the original human decision was wrong and the 
    # AI advice was correct, regardless of the final human decision, 
    # and zero otherwise.
    ca_cases = (
        df['user_base'] != df['true'] & 
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