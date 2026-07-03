#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import copy
import numpy as np
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt


DEFAULT_INVERSION_CONFIG = {
    "inversion_mode": "RGB", # "RGB" or "RG_ONLY"
    "minE0": 150,
    "secondorder": True,
    "generous": False,
    "plot": False,
    "checkagreement": True,
    "cutoffgoodness": 0.3,
    "q_interp_method": "linear", # "linear" or "nearest"
    "maxbluebright": "auto",
    "degen_bounds": None,
    "use_maglat_conductance_correction": True,
    "load_edens": True,
    "use_airglow": True,
}


def sig_integrator(sigmat, altvec, maglat=None, use_maglat_conductance_correction=True):
    """
    Height-integrate a 3D conductivity datacube to get a 2D conductance matrix.

    If use_maglat_conductance_correction=True, applies a first-order correction
    for magnetic field angle from vertical using maglat.
    """

    if hasattr(scipy.integrate, "cumulative_trapezoid"):
        sigmat_integrated = scipy.integrate.cumulative_trapezoid(
            sigmat,
            altvec / 100,
            axis=0,
        )[-1]
    else:
        sigmat_integrated = scipy.integrate.cumtrapz(
            sigmat,
            altvec / 100,
            axis=0,
        )[-1]

    if use_maglat_conductance_correction:
        if maglat is None:
            raise ValueError(
                "maglat is required when use_maglat_conductance_correction=True."
            )

        sigmat_integrated /= np.sin(maglat * np.pi / 180)

    return sigmat_integrated


def load_lookup_tables(fname_red, fname_green, fname_blue, fname_sigp, fname_sigh, maglat=None, fname_edens=None, use_maglat_conductance_correction=True, plot=False):
    """
    Read GLOW lookup tables and package them into a dictionary.
    """

    params, qvec, e0vec, greenmat = process_brightbin(fname_green, plot=plot)
    _, _, _, redmat = process_brightbin(fname_red, plot=plot)
    _, _, _, bluemat = process_brightbin(fname_blue, plot=plot)

    _, _, _, altvec, sigpmat = process_sig3dbin(fname_sigp)
    _, _, _, _, sighmat = process_sig3dbin(fname_sigh)

    sigp_integrated = sig_integrator(sigpmat, altvec, maglat=maglat, use_maglat_conductance_correction=use_maglat_conductance_correction)

    sigh_integrated = sig_integrator(sighmat, altvec, maglat=maglat, use_maglat_conductance_correction=use_maglat_conductance_correction)

    lookup_table = {
        "Params": params,
        "Qvec": qvec,
        "E0vec": e0vec,
        "greenmat": greenmat,
        "redmat": redmat,
        "bluemat": bluemat,
        "altvec": altvec,
        "sigPmat": sigpmat,
        "sigHmat": sighmat,
        "SigPmat": sigp_integrated,
        "SigHmat": sigh_integrated,
    }

    if fname_edens is not None:
        _, _, _, _, edensmat = process_sig3dbin(fname_edens)
        lookup_table["edensmat"] = edensmat

    return lookup_table


def load_lookup_tables_directory(folder, maglat=None, use_maglat_conductance_correction=True, load_edens=True, use_airglow=True, plot=False):
    """
    Read GLOW lookup tables from a directory.
    """

    fnamered = glob.glob(folder + "I6300*.bin")[0]
    fnamegreen = glob.glob(folder + "I5577*.bin")[0]
    fnameblue = glob.glob(folder + "I4278*.bin")[0]
    fnameped = glob.glob(folder + "ped3d*.bin")[0]
    fnamehall = glob.glob(folder + "hall3d*.bin")[0]

    fnameedens = None
    if load_edens:
        edens_matches = glob.glob(folder + "edens*.bin")
        if edens_matches:
            fnameedens = edens_matches[0]

    v = load_lookup_tables(fname_red=fnamered, fname_green=fnamegreen, fname_blue=fnameblue, 
                           fname_sigp=fnameped, fname_sigh=fnamehall, maglat=maglat, fname_edens=fnameedens, 
                           use_maglat_conductance_correction=use_maglat_conductance_correction, plot=plot)

    if use_airglow:
        fnamereda = glob.glob(folder + "airglow/" + "I6300*.bin")[0]
        fnamegreena = glob.glob(folder + "airglow/" + "I5577*.bin")[0]
        fnamebluea = glob.glob(folder + "airglow/" + "I4278*.bin")[0]
        fnamepeda = glob.glob(folder + "airglow/" + "ped3d*.bin")[0]
        fnamehalla = glob.glob(folder + "airglow/" + "hall3d*.bin")[0]

        fnameedensa = None
        if load_edens:
            edens_air_matches = glob.glob(folder + "airglow/" + "edens*.bin")
            if edens_air_matches:
                fnameedensa = edens_air_matches[0]

        va = load_lookup_tables(fname_red=fnamereda, fname_green=fnamegreena, fname_blue=fnamebluea, 
                                fname_sigp=fnamepeda, fname_sigh=fnamehalla, maglat=maglat, fname_edens=fnameedensa, 
                                use_maglat_conductance_correction=use_maglat_conductance_correction, plot=False)

        v["redbright_airglow"] = va["redmat"]
        v["bluebright_airglow"] = va["bluemat"]
        v["greenbright_airglow"] = va["greenmat"]

        v["sigP_bg"] = va["sigPmat"]
        v["sigH_bg"] = va["sigHmat"]

        v["SigP_bg"] = va["SigPmat"]
        v["SigH_bg"] = va["SigHmat"]

        if "edensmat" in va:
            v["edens_bg"] = va["edensmat"]

    else:
        v["redbright_airglow"] = np.zeros_like(v["redmat"])
        v["bluebright_airglow"] = np.zeros_like(v["bluemat"])
        v["greenbright_airglow"] = np.zeros_like(v["greenmat"])

        v["sigP_bg"] = np.zeros_like(v["sigPmat"])
        v["sigH_bg"] = np.zeros_like(v["sigHmat"])
        v["SigP_bg"] = np.zeros_like(v["SigPmat"])
        v["SigH_bg"] = np.zeros_like(v["SigHmat"])

    return v


def calculate_E0_Q_configurable(redbright, greenbright, bluebright, lookup_table, config=None):
    """
    Make config choices in top-level!!
    """

    cfg = DEFAULT_INVERSION_CONFIG.copy()

    if config is not None:
        cfg.update(config)

    mode = cfg.get("inversion_mode", "RGB")

    if mode == "RGB":
        return calculate_E0_Q_v2(
            redbright=redbright,
            greenbright=greenbright,
            bluebright=bluebright,
            inlookup_table=lookup_table,
            minE0=cfg.get("minE0", 150),
            secondorder=cfg.get("secondorder", True),
            generous=cfg.get("generous", False),
            q_interp_method=cfg.get("q_interp_method", "linear"),
            maxbluebright=cfg.get("maxbluebright", "auto"),
            degen_bounds=cfg.get("degen_bounds", None),
            plot=cfg.get("plot", False),
        )

    if mode == "RG_ONLY":
        qout, e0out = calculate_E0_Q_v2_RGonly(redbright=redbright, greenbright=greenbright, bluebright=bluebright,
                                               inlookup_table=lookup_table, minE0=cfg.get("minE0", 150), checkagreement=cfg.get("checkagreement", True),
                                               cutoffgoodness=cfg.get("cutoffgoodness", 0.3), generous=cfg.get("generous", False), plot=cfg.get("plot", False))

        nan_uncertainty = np.full_like(qout, np.nan)

        return (qout, e0out, nan_uncertainty, nan_uncertainty, nan_uncertainty, nan_uncertainty)

    raise ValueError(f"Unknown inversion_mode: {mode}")


def calculate_E0_Q_v2(
    redbright,
    greenbright,
    bluebright,
    inlookup_table,
    minE0=150,
    secondorder=True,
    generous=False,
    q_interp_method="linear",
    maxbluebright="auto",
    degen_bounds=None,
    plot=False,
):
    """
    RGB inversion.

    Uses blue brightness to estimate Q, then red/green ratio to estimate E0.
    Optionally performs a second constrained Q fit using first-pass E0 bounds.
    """

    lookup_table = copy.deepcopy(inlookup_table)

    lookup_table["redmat"] -= lookup_table["redbright_airglow"][0][0]
    lookup_table["greenmat"] -= lookup_table["greenbright_airglow"][0][0]
    lookup_table["bluemat"] -= lookup_table["bluebright_airglow"][0][0]

    shape = greenbright.shape

    redvec = redbright.reshape(-1)
    greenvec = greenbright.reshape(-1)
    bluevec = bluebright.reshape(-1)

    minE0ind = np.where(lookup_table["E0vec"] > minE0)[0][0]

    qvec, maxqvec, minqvec = q_interp(
        lookup_table["bluemat"],
        lookup_table["Qvec"],
        lookup_table["E0vec"],
        bluevec,
        minE0ind=minE0ind,
        maxbluebright=maxbluebright,
        interp=q_interp_method,
        plot=plot,
    )

    e0vec = e0_interp_general(
        lookup_table["redmat"] / lookup_table["greenmat"],
        lookup_table["Qvec"],
        lookup_table["E0vec"],
        redvec / greenvec,
        qvec,
        generous=generous,
        degen_bounds=degen_bounds,
    )

    e0vecext1 = e0_interp_general(
        lookup_table["redmat"] / lookup_table["greenmat"],
        lookup_table["Qvec"],
        lookup_table["E0vec"],
        redvec / greenvec,
        maxqvec,
        generous=generous,
        degen_bounds=degen_bounds,
    )

    e0vecext2 = e0_interp_general(
        lookup_table["redmat"] / lookup_table["greenmat"],
        lookup_table["Qvec"],
        lookup_table["E0vec"],
        redvec / greenvec,
        minqvec,
        generous=generous,
        degen_bounds=degen_bounds,
    )

    mine0vec = np.minimum(e0vecext1, e0vecext2)
    maxe0vec = np.maximum(e0vecext1, e0vecext2)

    if secondorder:
        qvec, maxqvec, minqvec = q_interp_constrained(
            lookup_table["bluemat"],
            mine0vec,
            maxe0vec,
            lookup_table["Qvec"],
            lookup_table["E0vec"],
            bluevec,
            backupminE0ind=minE0ind,
            interp=q_interp_method,
            plot=plot,
        )

        e0vec = e0_interp_general(
            lookup_table["redmat"] / lookup_table["greenmat"],
            lookup_table["Qvec"],
            lookup_table["E0vec"],
            redvec / greenvec,
            qvec,
            generous=generous,
            degen_bounds=degen_bounds,
        )

        e0vecext1 = e0_interp_general(
            lookup_table["redmat"] / lookup_table["greenmat"],
            lookup_table["Qvec"],
            lookup_table["E0vec"],
            redvec / greenvec,
            maxqvec,
            generous=generous,
            degen_bounds=degen_bounds,
        )

        e0vecext2 = e0_interp_general(
            lookup_table["redmat"] / lookup_table["greenmat"],
            lookup_table["Qvec"],
            lookup_table["E0vec"],
            redvec / greenvec,
            minqvec,
            generous=generous,
            degen_bounds=degen_bounds,
        )

        mine0vec = np.minimum(e0vecext1, e0vecext2)
        maxe0vec = np.maximum(e0vecext1, e0vecext2)

    if generous:
        qvec[np.where(bluevec < np.amin(lookup_table["bluemat"]))] = 0
        e0vec[np.where((redvec / greenvec) > np.amax(lookup_table["redmat"] / lookup_table["greenmat"]))] = 0

    return (
        qvec.reshape(shape),
        e0vec.reshape(shape),
        minqvec.reshape(shape),
        maxqvec.reshape(shape),
        mine0vec.reshape(shape),
        maxe0vec.reshape(shape),
    )


def calculate_E0_Q_v2_RGonly(
    redbright,
    greenbright,
    bluebright,
    inlookup_table,
    minE0=150,
    checkagreement=True,
    cutoffgoodness=0.3,
    generous=False,
    plot=False,
):
    """
    Estimate Q and E0 without explicitly using blue brightness as the primary
    Q constraint.

    This still accepts bluebright because the optional GLOW agreement check uses
    RGB consistency.
    """

    lookup_table = copy.deepcopy(inlookup_table)

    lookup_table["redmat"] -= lookup_table["redbright_airglow"][0][0]
    lookup_table["greenmat"] -= lookup_table["greenbright_airglow"][0][0]
    lookup_table["bluemat"] -= lookup_table["bluebright_airglow"][0][0]

    shape = greenbright.shape

    redvec = redbright.reshape(-1)
    greenvec = greenbright.reshape(-1)

    _ = np.where(lookup_table["E0vec"] > minE0)[0][0]

    qmat = q_interp_RG(
        lookup_table["redmat"],
        lookup_table["greenmat"],
        lookup_table["bluemat"],
        lookup_table["Qvec"],
        lookup_table["E0vec"],
        redbright,
        greenbright,
        bluebright,
        checkagreement=checkagreement,
        cutoffgoodness=cutoffgoodness,
        plot=plot,
    )

    qvec = np.copy(qmat).reshape(-1)

    e0vec = e0_interp_general(
        lookup_table["redmat"] / lookup_table["greenmat"],
        lookup_table["Qvec"],
        lookup_table["E0vec"],
        redvec / greenvec,
        qvec,
        generous=generous,
    )

    if generous:
        e0vec[np.where((redvec / greenvec) > np.amax(lookup_table["redmat"] / lookup_table["greenmat"]))] = 0

    return qmat, e0vec.reshape(shape)


def calculate_E0_Q(redbright, greenbright, bluebright, lookup_table, minE0=150, generous=False):
    """
    For calls with the older config files...like v1/v2/v3 that are kinda obsolete
    """

    return calculate_E0_Q_v2(redbright=redbright,
        greenbright=greenbright,
        bluebright=bluebright,
        inlookup_table=lookup_table,
        minE0=minE0,
        secondorder=False,
        generous=generous,
        q_interp_method="linear",
        maxbluebright="auto",
        plot=False,
    )


def calculate_Sig(q, e0, lookup_table, generous=False):
    """
    Given Q and E0 arrays, interpolate to calculate Pedersen and Hall conductances.
    """

    shape = q.shape

    qvec = q.reshape(-1)
    e0vec = e0.reshape(-1)

    sigp_interp = scipy.interpolate.RegularGridInterpolator(
        [lookup_table["E0vec"], lookup_table["Qvec"]],
        lookup_table["SigPmat"],
        bounds_error=False,
        fill_value=np.nan,
    )

    sigh_interp = scipy.interpolate.RegularGridInterpolator(
        [lookup_table["E0vec"], lookup_table["Qvec"]],
        lookup_table["SigHmat"],
        bounds_error=False,
        fill_value=np.nan,
    )

    sigpout = np.zeros_like(qvec)
    sighout = np.zeros_like(qvec)

    nancond = np.isnan(qvec) | np.isnan(e0vec)
    lessercond = (
        (qvec < np.amin(lookup_table["Qvec"]))
        | (e0vec < np.amin(lookup_table["E0vec"]))
    )
    greatercond = (
        (qvec > np.amax(lookup_table["Qvec"]))
        | (e0vec > np.amax(lookup_table["E0vec"]))
    )

    zerocond = (qvec == 0) | (e0vec == 0)

    mask = nancond | lessercond | greatercond | zerocond

    invec = np.asarray([e0vec[np.where(~mask)], qvec[np.where(~mask)]]).T

    sigpout[np.where(mask)] = np.nan
    sighout[np.where(mask)] = np.nan

    if len(invec) > 0:
        sigpout[np.where(~mask)] = sigp_interp(invec)
        sighout[np.where(~mask)] = sigh_interp(invec)

    if generous:
        if "SigP_bg" in lookup_table:
            sigpout[np.where(lessercond | zerocond)] = np.nanmean(lookup_table["SigP_bg"])
        else:
            sigpout[np.where(lessercond | zerocond)] = np.nanmin(sigpout)

        if "SigH_bg" in lookup_table:
            sighout[np.where(lessercond | zerocond)] = np.nanmean(lookup_table["SigH_bg"])
        else:
            sighout[np.where(lessercond | zerocond)] = np.nanmin(sighout)

    return sigpout.reshape(shape), sighout.reshape(shape)


def process_brightbin(fname, plot=False):
    """
    Process one GLOW brightness lookup table.
    """

    with open(fname) as file_obj:
        recs = np.fromfile(file_obj, dtype="float32")

    params = recs[:20]

    nq = int(recs[20])
    ne = int(recs[21])

    qvec = recs[22: 22 + nq]
    e0vec = recs[22 + nq: 22 + nq + ne]

    bright = recs[22 + nq + ne:].reshape(ne, nq)

    if plot:
        plt.pcolormesh(qvec, e0vec, bright, shading="auto")
        plt.xlabel("Q")
        plt.ylabel("E0")
        plt.title(fname.split("/")[-1][1:5])
        plt.colorbar()
        plt.show()

    return params, qvec, e0vec, bright


def process_sig3dbin(fname):
    """
    Process one GLOW 3D binary table: conductance or electron density.
    """

    with open(fname) as file_obj:
        recs = np.fromfile(file_obj, dtype="float32")

    params = recs[:20]

    nq = int(recs[20])
    ne = int(recs[21])
    nalt = int(recs[22])

    qvec = recs[23: 23 + nq]
    e0vec = recs[23 + nq: 23 + nq + ne]
    altvec = recs[23 + nq + ne: 23 + nq + ne + nalt]

    sig3d = recs[23 + nq + ne + nalt:].reshape(nalt, ne, nq)

    return params, qvec, e0vec, altvec, sig3d


def q_interp(
    bright428,
    qvec_in,
    e0vec,
    bluevec,
    minE0ind=0,
    maxbluebright="auto",
    interp="linear",
    plot=False,
):
    """
    Use blue-line brightness to estimate Q.
    """

    if maxbluebright == "auto":
        testbluevec = np.linspace(0, np.amax(bright428), 50)

        _, testmaxqvec, _ = q_interp(
            bright428,
            qvec_in,
            e0vec,
            testbluevec,
            minE0ind=minE0ind,
            maxbluebright=np.inf,
            interp=interp,
            plot=False,
        )

        good = np.where(~np.isnan(testmaxqvec))[0]

        if len(good) > 2:
            medval = np.median(np.diff(testmaxqvec[good]))
            bad = np.where(np.diff(testmaxqvec) < (medval / 2))[0]

            if len(bad) > 0:
                maxbluebright = testbluevec[bad[0]]
            else:
                maxbluebright = np.inf
        else:
            maxbluebright = np.inf

        if plot:
            plt.scatter(testbluevec, testmaxqvec, color="black")
            plt.title("Max possible Q")
            plt.xlabel("Blue brightness")
            plt.ylabel("Max Q")
            plt.show()

    qout = []
    maxqout = []
    minqout = []

    if plot:
        plt.pcolormesh(qvec_in, e0vec, bright428, shading="auto")
        plt.xlabel("Q")
        plt.ylabel("E0")

    for blue in bluevec:
        if (blue < np.amin(bright428)) or (blue > np.amax(bright428)):
            qout.append(np.nan)
            maxqout.append(np.nan)
            minqout.append(np.nan)
            continue

        qcross = []
        e0cross = []

        for e0i in range(minE0ind, len(e0vec)):
            try:
                crossings = np.where(
                    np.diff(np.sign(bright428[e0i, :] - blue))
                )[0]

                if len(crossings) == 0:
                    continue

                guessind = crossings[0]

                if interp == "nearest":
                    qi = qvec_in[guessind]

                elif interp == "linear":
                    slope = (
                        bright428[e0i, guessind + 1]
                        - bright428[e0i, guessind]
                    ) / (
                        qvec_in[guessind + 1]
                        - qvec_in[guessind]
                    )

                    qi = (
                        (blue - bright428[e0i, guessind]) / slope
                        + qvec_in[guessind]
                    )

                else:
                    raise ValueError(f"Unknown q_interp interp mode: {interp}")

                qcross.append(qi)
                e0cross.append(e0vec[e0i])

            except Exception:
                pass

        qcross = np.asarray(qcross)

        if len(qcross) > 0:
            qout.append(qcross[-1])

            if blue > maxbluebright:
                maxqout.append(np.nan)
            else:
                maxqout.append(np.amax(qcross))

            minqout.append(np.amin(qcross))

            if plot:
                plt.plot(qcross, e0cross)
                plt.scatter(qcross[-1], e0cross[-1], color="black", s=50)

        else:
            qout.append(np.nan)
            maxqout.append(np.nan)
            minqout.append(np.nan)

    if plot:
        plt.plot(
            [qvec_in[0], qvec_in[-1]],
            [e0vec[minE0ind], e0vec[minE0ind]],
            color="black",
            linewidth=5,
        )
        plt.title("Solution sets for blue-line brightness")
        plt.show()

    return np.asarray(qout), np.asarray(maxqout), np.asarray(minqout)


def q_interp_constrained(
    bright428,
    minE0vec,
    maxE0vec,
    qvec_in,
    e0vec,
    bluevec,
    backupminE0ind=0,
    interp="linear",
    plot=False,
):
    """
    Same as q_interp, but constrain possible Q using E0 bounds.
    """

    qout = []
    maxqout = []
    minqout = []

    if plot:
        plt.pcolormesh(qvec_in, e0vec, bright428, shading="auto")
        plt.xlabel("Q")
        plt.ylabel("E0")

    for i, blue in enumerate(bluevec):
        if np.isnan(minE0vec[i]):
            minE0ind = backupminE0ind
        else:
            min_candidates = np.where(e0vec <= minE0vec[i])[0]
            minE0ind = min_candidates[-1] if len(min_candidates) else backupminE0ind

        if np.isnan(maxE0vec[i]):
            maxE0ind = len(e0vec) - 1
        else:
            max_candidates = np.where(e0vec >= maxE0vec[i])[0]
            maxE0ind = max_candidates[0] if len(max_candidates) else len(e0vec) - 1

        qcross = []
        e0cross = []

        for e0i in range(minE0ind, maxE0ind + 1):
            try:
                crossings = np.where(
                    np.diff(np.sign(bright428[e0i, :] - blue))
                )[0]

                if len(crossings) == 0:
                    continue

                guessind = crossings[0]

                if interp == "nearest":
                    qi = qvec_in[guessind]

                elif interp == "linear":
                    slope = (
                        bright428[e0i, guessind + 1]
                        - bright428[e0i, guessind]
                    ) / (
                        qvec_in[guessind + 1]
                        - qvec_in[guessind]
                    )

                    qi = (
                        (blue - bright428[e0i, guessind]) / slope
                        + qvec_in[guessind]
                    )

                else:
                    raise ValueError(f"Unknown q_interp_constrained interp mode: {interp}")

                qcross.append(qi)
                e0cross.append(e0vec[e0i])

            except Exception:
                pass

        qcross = np.asarray(qcross)

        if len(qcross) > 0:
            qout.append(np.median(qcross))

            if (
                not np.isnan(minE0vec[i])
                and not np.isnan(maxE0vec[i])
                and minE0vec[i] >= e0vec[0]
                and maxE0vec[i] <= e0vec[-1]
            ):
                maxqout.append(np.amax(qcross))
                minqout.append(np.amin(qcross))
            else:
                maxqout.append(np.nan)
                minqout.append(np.nan)

            if plot:
                plt.plot(qcross, e0cross)
                plt.scatter(qcross[-1], e0cross[-1], color="black", s=50)

        else:
            qout.append(np.nan)
            maxqout.append(np.nan)
            minqout.append(np.nan)

    if plot:
        plt.plot(
            [qvec_in[0], qvec_in[-1]],
            [e0vec[backupminE0ind], e0vec[backupminE0ind]],
            color="black",
            linewidth=5,
        )
        plt.title("Constrained Q solution sets")
        plt.show()

    return np.asarray(qout), np.asarray(maxqout), np.asarray(minqout)


def q_interp_RG(
    bright630,
    bright558,
    bright428,
    qvec_in,
    e0vec,
    redbright,
    greenbright,
    bluebright,
    checkagreement=True,
    cutoffgoodness=0.3,
    plot=False,
):
    """
    Estimate Q from red and green brightness without using blue brightness as
    the main constraint.

    Blue may still be used to check RGB agreement with GLOW.
    """

    def newcoords(r, g, b):
        newx = np.log(g) - np.log(r)
        newy = np.log(g) - np.log(b)
        newz = np.log(r) + np.log(g) + np.log(b)
        return newx, newy, newz

    valid_data = np.where(
        ~np.isnan((redbright + greenbright + bluebright).reshape(-1))
    )[0]

    if len(valid_data) == 0:
        return np.full_like(redbright, np.nan)

    redmax = np.nanmax(redbright.reshape(-1)[valid_data])
    greenmax = np.nanmax(greenbright.reshape(-1)[valid_data])
    bluemax = np.nanmax(bluebright.reshape(-1)[valid_data])

    redin = bright630.reshape(-1)
    greenin = bright558.reshape(-1)
    bluein = bright428.reshape(-1)

    goodrange = np.where(
        (redin < redmax)
        & (greenin < greenmax)
        & (bluein < bluemax)
        & (redin > 0)
        & (greenin > 0)
        & (bluein > 0)
    )[0]

    if len(goodrange) == 0:
        return np.full_like(redbright, np.nan)

    if len(goodrange) > 20000:
        dec = int(np.ceil(np.sqrt(len(goodrange) / 20000)))

        redin_q = bright630[::dec, ::dec].reshape(-1)
        greenin_q = bright558[::dec, ::dec].reshape(-1)
        bluein_q = bright428[::dec, ::dec].reshape(-1)

        goodrange = np.where(
            (redin_q < redmax)
            & (greenin_q < greenmax)
            & (bluein_q < bluemax)
            & (redin_q > 0)
            & (greenin_q > 0)
            & (bluein_q > 0)
        )[0]

        qmat, e0mat = np.meshgrid(qvec_in[::dec], e0vec[::dec])

    else:
        redin_q = bright630.reshape(-1)
        greenin_q = bright558.reshape(-1)
        bluein_q = bright428.reshape(-1)
        qmat, e0mat = np.meshgrid(qvec_in, e0vec)

    redplot = redin_q[goodrange]
    greenplot = greenin_q[goodrange]
    blueplot = bluein_q[goodrange]

    qplot = qmat.reshape(-1)[goodrange]

    red_data = np.copy(redbright)
    green_data = np.copy(greenbright)
    blue_data = np.copy(bluebright)

    red_data[red_data <= 0] = np.nan
    green_data[green_data <= 0] = np.nan
    blue_data[blue_data <= 0] = np.nan

    try:
        qinterp = scipy.interpolate.RBFInterpolator(
            np.asarray([np.log(redplot), np.log(greenplot)]).T,
            qplot,
            kernel="cubic",
        )

        qdata = qinterp(
            np.asarray(
                [
                    np.log(red_data).reshape(-1),
                    np.log(green_data).reshape(-1),
                ]
            ).T
        ).reshape(redbright.shape)

    except Exception:
        return np.full_like(redbright, np.nan)

    if plot and len(qdata.shape) > 1:
        plt.pcolormesh(qdata, vmin=0, vmax=12)
        plt.colorbar()
        plt.title("Q from log(red), log(green)")
        plt.show()

    if checkagreement:
        newx, newy, _ = newcoords(redplot, greenplot, blueplot)
        newxdata, newydata, _ = newcoords(red_data, green_data, blue_data)

        nbins = 35
        newxs0 = np.linspace(np.nanmin(newx), np.nanmax(newx), nbins)
        newys = np.zeros(len(newxs0) - 1)

        for i in range(len(newys)):
            xrange = np.where((newx >= newxs0[i]) & (newx < newxs0[i + 1]))[0]

            if len(xrange) > 0:
                newys[i] = np.nanmedian(newy[xrange])
            else:
                newys[i] = np.nan

        newxs = np.asarray(
            [
                (newxs0[i] + newxs0[i + 1]) / 2
                for i in range(len(newxs0) - 1)
            ]
        )

        good = np.where(~np.isnan(newys))[0]

        if len(good) > 4:
            newxs = newxs[good]
            newys = newys[good]

            cs = scipy.interpolate.CubicSpline(newxs, newys)

            xcurve = np.linspace(np.nanmin(newxs), np.nanmax(newxs), 1000)
            ycurve = cs(xcurve)

            newxfordiffvec = np.linspace(np.nanmin(xcurve) - 2, np.nanmax(xcurve) + 2, 50)
            newyfordiffvec = np.linspace(np.nanmin(ycurve) - 2, np.nanmax(ycurve) + 2, 51)

            newyfordiff, newxfordiff = np.meshgrid(
                newyfordiffvec,
                newxfordiffvec,
            )

            distmat = np.zeros_like(newxfordiff)

            for i in range(len(newxfordiffvec)):
                for j in range(len(newyfordiffvec)):
                    distmat[i, j] = np.nanmin(
                        (ycurve - newyfordiff[i, j]) ** 2
                        + (xcurve - newxfordiff[i, j]) ** 2
                    )

            distinterp = scipy.interpolate.RBFInterpolator(
                np.asarray(
                    [
                        newxfordiff.reshape(-1),
                        newyfordiff.reshape(-1),
                    ]
                ).T,
                distmat.reshape(-1),
                kernel="cubic",
            )

            distdata = distinterp(
                np.asarray(
                    [
                        newxdata.reshape(-1),
                        newydata.reshape(-1),
                    ]
                ).T
            ).reshape(newxdata.shape)

            badrange = np.where(
                ((1 - distdata) < cutoffgoodness)
                | np.isnan(blue_data + distdata)
            )

            qdata[badrange] = np.nan

            if plot:
                plt.pcolormesh(1 - distdata, vmin=cutoffgoodness, cmap="plasma")
                plt.title("Goodness of agreement with GLOW")
                plt.colorbar()
                plt.show()

    return qdata


def e0_interp_general(
    testmat,
    qvec_in,
    e0vec,
    testvec,
    qinvec,
    generous=False,
    degen_bounds=None,
):
    """
    Interpolate E0 from a general lookup table F(Q, E0).
    """

    e0out = []

    indvec = []

    for qin in qinvec:
        try:
            indvec.append(np.where(qvec_in < qin)[0][-1])
        except Exception:
            indvec.append(np.nan)

    for i in range(len(testvec)):
        if np.isnan(indvec[i]):
            e0out.append(np.nan)
            continue

        ind = int(indvec[i])

        if ind == len(qvec_in) - 1:
            if generous:
                curve = np.copy(testmat[:, -1]) - testvec[i]
            else:
                e0out.append(np.nan)
                continue

        else:
            fracind = (qinvec[i] - qvec_in[ind]) / (qvec_in[ind + 1] - qvec_in[ind])
            curve0 = (1 - fracind) * testmat[:, ind] + fracind * testmat[:, ind + 1]
            curve = np.copy(curve0) - testvec[i]

        try:
            crossings = np.where(np.diff(np.sign(curve)))[0]
            guessind = crossings[0]

            if len(crossings) > 1:
                if (len(crossings) > 2) or (np.diff(crossings)[0] != 1):
                    if degen_bounds is None:
                        guessind = np.nan
                    else:
                        crossings_mask = (
                            (e0vec[crossings] > degen_bounds[-1])
                            | (e0vec[crossings] < degen_bounds[0])
                        )

                        if len(crossings[~crossings_mask]) == 1:
                            guessind = crossings[~crossings_mask][0]
                        else:
                            guessind = np.nan

            slope = (curve[guessind + 1] - curve[guessind]) / (
                e0vec[guessind + 1] - e0vec[guessind]
            )

            e0i = -curve[guessind] / slope + e0vec[guessind]

        except Exception:
            e0i = np.nan

        e0out.append(e0i)

    return np.asarray(e0out)


def e0_interp_general_nearest(testmat, qvec_in, e0vec, testvec, qinvec):
    """
    Nearest-neighbor E0 interpolation.
    """

    e0out = []
    indvec = np.asarray([np.argmin(np.abs(qvec_in - qin)) for qin in qinvec])

    for i in range(len(testvec)):
        if np.isnan(indvec[i]):
            e0out.append(np.nan)
            continue

        curve = testmat[:, indvec[i]] - testvec[i]

        try:
            e0i = e0vec[np.where(np.diff(np.sign(curve)))[0][0]]
        except Exception:
            e0i = np.nan

        e0out.append(e0i)

    return np.asarray(e0out)
