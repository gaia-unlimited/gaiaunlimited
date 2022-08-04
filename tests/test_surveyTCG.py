from gaiasf.selectionfunctions.surveyTCG import DR3SelectionFunctionTCG


def test_tcg():

    try:
        DR3SelectionFunctionTCG()
    except:
        assert False, "Failed"
