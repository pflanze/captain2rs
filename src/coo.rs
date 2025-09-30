pub struct Coo4 {
    x: Vec<u16>,
    y: Vec<u16>,
    z: Vec<u16>,
    i: Vec<u16>,
    val: Vec<f64>,
}

impl Coo4 {
    pub fn new() -> Self {
        Self {
            x: Vec::new(),
            y: Vec::new(),
            z: Vec::new(),
            i: Vec::new(),
            val: Vec::new(),
        }
    }

    pub fn insert(&mut self, (x, y, z, i): (u16, u16, u16, u16), val: f64) {
        self.x.push(x);
        self.y.push(y);
        self.z.push(z);
        self.i.push(i);
        self.val.push(val);
    }

    pub fn len(&self) -> usize {
        self.val.len()
    }
}
