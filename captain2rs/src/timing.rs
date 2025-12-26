use std::time::SystemTime;

pub struct Timing {
    pub now: SystemTime,
    pub span_name: &'static str,
}

pub fn show_current_timing(
    show: bool,
    last_timing: Option<Timing>,
    span_name: &'static str,
) -> Option<Timing> {
    if show {
        let now = SystemTime::now();
        if let Some(last_timing) = last_timing {
            let dur = now
                .duration_since(last_timing.now)
                .expect("always increasing");
            eprintln!("timing: {}: {} s", last_timing.span_name, dur.as_secs_f64());
        }
        Some(Timing { now, span_name })
    } else {
        None
    }
}
