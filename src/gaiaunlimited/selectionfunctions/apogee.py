from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as units
from astropy.table import Table
import numpy as np
from pathlib import Path
import pkg_resources

__all__ = ["apogee_sf"]


def apogee_sf(apparentH, unreddenedJK, position):
    """
    Returns the probability that a star of given intrinsic colour (J-K)_0
    and apparent magnitude H and given position on the sky was targeted by
    APOGEE or APOGEE-2.

    Args:
        apparentH (float or 1-d array): the apparent magnitude H of the source.

        unreddenedJK (float or 1-d array): the intrinsic (J-K) colour of the source.

        position (:obj:`astropy.coordinates.SkyCoord`): Sky coordinates at which
            to compute the selection function.

    Returns:
        The selection function as either a float (between 0 and 1) or a numpy array
        of floats, with the same length as the input.
    """
    # Read the precomputed table with the details for each field:
    apogee_frac_path = pkg_resources.resource_filename(
        "gaiaunlimited", "data/apogee_sampling_fractions.csv"
    )
    if not Path(apogee_frac_path).resolve().is_file():
        datadir_path = pkg_resources.resource_filename("gaiaunlimited", "data/")
        import os

        print(datadir_path)
        print(os.listdir(datadir_path))
        print(os.listdir(datadir_path + "/.."))
        print(Path(apogee_frac_path).resolve())
        raise ValueError("Precomputed APOGEE selection fraction file not found.")
    tApogeeSF = Table.read(apogee_frac_path)
    allFieldNames = sorted(set(tApogeeSF["FIELD"]))
    allFieldCenterL = [
        tApogeeSF["GLON"][tApogeeSF["FIELD"] == f][0] for f in allFieldNames
    ]
    allFieldCenterB = [
        tApogeeSF["GLAT"][tApogeeSF["FIELD"] == f][0] for f in allFieldNames
    ]
    allFieldNames = np.array(
        [tApogeeSF["FIELD"][tApogeeSF["FIELD"] == f][0] for f in allFieldNames]
    )
    allFieldRadii = np.array(
        [tApogeeSF["RADIUS"][tApogeeSF["FIELD"] == f][0] for f in allFieldNames]
    )
    coord_centres = SkyCoord(
        l=allFieldCenterL * units.deg, b=allFieldCenterB * units.deg, frame="galactic"
    )

    try:
        nbStars = len(apparentH)
        inputIsList = True
    except:
        nbStars = 1
        # the input is a single star
        apparentH = [apparentH]
        unreddenedJK = [unreddenedJK]
        position = [position]
        inputIsList = False

    to_return = []

    for i in range(len(apparentH)):
        # For each star we store the selection fraction and the A_K fraction
        # of each field it belongs to.
        selectionFraction = []
        akFraction = []

        if apparentH[i] < 5 or apparentH[i] > 13.8:
            # we know these stars are never picked for targeting
            to_return.append(0)
            continue

        sep = position[i].separation(coord_centres)
        fieldOfSource = allFieldNames[sep.deg < allFieldRadii]

        if len(fieldOfSource) == 0:
            # this location is not covered by any APOGEE(-2) grid pointing
            to_return.append(0)
            continue

        # Now we know the source is inside at least one APOGEE(-2) pointing
        # The vast majority of the sky is only covered by one pointing,
        # but we still need to handle the rare cases, so we iterate over the
        # pointings the source is a part of:
        for f in fieldOfSource:
            tF = tApogeeSF[tApogeeSF["FIELD"] == f]

            if tF["cohort"][0] == "short_blue":
                # this is an APOGEE-2 field with colour binning
                # we need to find out if our source is considered blue or red
                JKlimit = tF["JK0max"][0]
                if unreddenedJK[i] < JKlimit:
                    tF = tF[[0, 2, 4]]
                else:
                    tF = tF[[1, 3, 5]]

            if apparentH[i] < tF["Hmin"][0] or apparentH[i] > tF["Hmax"][2]:
                # the star is too bright or too faint
                selectionFraction.append(0)
                akFraction.append(0)

            # bright enough to be short cohort:
            elif apparentH[i] < tF["Hmax"][0]:
                # short cohort
                selectionFraction.append(tF["fracSampling"][0])
                akFraction.append(tF["fracAK"][0])

            # medium cohort
            elif apparentH[i] < tF["Hmax"][1] and apparentH[i] > tF["Hmin"][1]:
                # medium cohort
                selectionFraction.append(tF["fracSampling"][1])
                akFraction.append(tF["fracAK"][1])

            # long cohort
            elif apparentH[i] < tF["Hmax"][2] and apparentH[i] > tF["Hmin"][2]:
                # long cohort
                selectionFraction.append(tF["fracSampling"][2])
                akFraction.append(tF["fracAK"][2])

            #not in any cohort: typically between short/medium in some fields
            else:    
                selectionFraction.append(0)
                akFraction.append(0)

        if len(selectionFraction) == 1:
            to_return.append(selectionFraction[0] * akFraction[0])
        else:
            to_return.append(
                (
                    sum(selectionFraction)
                    - np.prod(selectionFraction) * 0.5 * sum(akFraction)
                )
            )

    if inputIsList:
        return to_return
    else:
        return to_return[0]
