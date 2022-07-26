from gaiasf.selectionfunctions.surveyTC import DR3SelectionFunctionTCG


def test_tcg():

    try:
        DR3SelectionFunctionTCG()
    except:
        assert False, "Failed"
