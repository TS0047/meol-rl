# fg_replay.py and fg_play.py run FlightGear with --fdm=null and drive the jet over native-fdm, so the
# F-16 addon's JSBSim properties never exist: its HUD shows no speed, FLYUP stays lit, and with no
# Mach the pitot reads zero. Every frame, this copies what native-fdm did set into the properties
# the F-16 reads.

var J = "/fdm/jsbsim/";
var FPS2KT = 0.592484;
var timer = nil;
var hud_timer = nil;

var feed = func {
    var u = getprop("/velocities/uBody-fps") or 0;
    var v = getprop("/velocities/vBody-fps") or 0;
    var w = getprop("/velocities/wBody-fps") or 0;
    var vt = math.sqrt(u * u + v * v + w * w);
    var sound_kt = 38.967854 * math.sqrt((getprop("/environment/temperature-degc") or 15) + 273.15);
    setprop("/velocities/mach", vt * FPS2KT / sound_kt);          # the pitot, so the airspeed gauge reads
    setprop(J ~ "velocities/vc-kts", getprop("/velocities/airspeed-kt") or 0);
    setprop(J ~ "velocities/vt-fps", vt);
    setprop(J ~ "velocities/vtrue-kts", vt * FPS2KT);
    setprop(J ~ "velocities/v-fps", v);
    setprop(J ~ "velocities/w-fps", w);
    setprop(J ~ "aero/alpha-deg", getprop("/orientation/alpha-deg") or 0);
    setprop(J ~ "aero/beta-deg", getprop("/orientation/side-slip-deg") or 0);
    # the F-16's own time-to-impact loop needs its JSBSim (electrics, fuel, aero) and never runs here;
    # the HUD lights FLYUP under 8 s, so give it seconds to the sea at the current sink rate
    var sink = getprop("/velocities/speed-down-fps") or 0;
    var agl = getprop("/position/altitude-agl-ft") or 0;
    setprop("/instrumentation/radar/time-till-crash", sink > 1 ? math.min(15, agl / sink) : 15);
};

# The F-16's init (f16.nas) stops in its first loop, which needs that JSBSim, before it registers
# the HUD and starts the frame-notification loop that redraws it: do both once the aircraft is up.
var STATUS = "/addons/meol/hud";                                  # read by fg_*.py over telnet
var hud_up = func {
    var ns = contains(globals, "f16") ? globals["f16"] : nil;
    if (ns == nil or !contains(ns, "f16_hud") or !contains(ns, "rtExec_loop"))
        return setprop(STATUS, "not the F-16");
    setprop("/controls/HUD/display-on", 1);                       # HUD power: the F-16's electrics,
    setprop("/f16/avionics/hud-power", 1);                        # which need its JSBSim
    setprop("/f16/avionics/hud-brt", 1);
    if (!getprop("/f16/avionics/hud-sym"))
        setprop("/f16/avionics/hud-sym", 1);                      # symbology brightness knob
    var err = [];
    call(func {
        emesary.GlobalTransmitter.Register(ns.f16_hud);
        ns.rtExec_loop();
    }, nil, nil, nil, err);
    setprop(STATUS, size(err) ? "failed: " ~ err[0] : "registered");
};

var main = func(addon) {
    if (getprop("/sim/flight-model") != "null")
        return;                                                   # a real FDM provides all of this
    foreach (var p; ["atmosphere/density-altitude", "fcs/flap-pos-deg", "systems/approach-speed",
                     "velocities/r-rad_sec", "autoflight/pitch/vs/target-kin"])
        setprop(J ~ p, 0);                                        # the rest of what the HUD reads
    hud_timer = maketimer(2, hud_up);                             # add-ons may not use settimer()
    hud_timer.singleShot = 1;
    if (getprop("/sim/signals/fdm-initialized"))                  # add-ons usually load after it
        hud_timer.start();
    else
        setlistener("/sim/signals/fdm-initialized", func(n) {
            if (n.getBoolValue())
                hud_timer.start();                                # after the F-16's own init has run
        }, 0, 0);
    # gear up and in the air; set before the F-16's own init, whose loop never starts if these are missing
    setprop(J ~ "gear/gear-pos-norm", 0);
    setprop(J ~ "gear/wow", 0);
    for (var i = 0; i < 3; i += 1) {
        setprop(J ~ "gear/unit[" ~ i ~ "]/WOW", 0);
        setprop("/gear/gear[" ~ i ~ "]/position-norm", 0);
    }
    timer = maketimer(0, feed);                                   # every frame
    timer.start();
    logprint(LOG_INFO, "MEOL native-fdm instruments: feeding the F-16 HUD");
};
