def test_import_hmle():
	try:
		from gaiaunlimited import subsample
		subsampleSF_HMLE = subsample.SubsampleSelectionFunctionHMLE()
		success = True
	except:
		success = False
	assert success