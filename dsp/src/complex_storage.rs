use crate::{normalize_sc16_q11, num_complex::Complex};

// TODO - VecDeque
pub struct ComplexStorage {
    buffer: Vec<Complex<f64>>,
}

impl ComplexStorage {
    pub fn new(initial_capacity: usize) -> Self {
        ComplexStorage {
            buffer: Vec::with_capacity(initial_capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    // Samples.len() must be a multiple of 2, converts I/Q pair to complex
    pub fn push_normalize_sc16_q11(&mut self, samples: &[i16]) {
        debug_assert!(samples.len() % 2 == 0);
        samples.chunks(2).for_each(|pair| {
            let (i, q) = (normalize_sc16_q11(pair[0]), normalize_sc16_q11(pair[1]));
            self.buffer.push(Complex::new(i, q));
        });
    }

    pub fn buffer(&mut self) -> &[Complex<f64>] {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut [Complex<f64>] {
        &mut self.buffer
    }

    pub fn drain(&mut self, size: usize) {
        self.buffer.drain(..size);
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_storage_drains() {
        let mut s = ComplexStorage::new(10);
        assert_eq!(s.len(), 0);
        s.push_normalize_sc16_q11(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(s.len(), 3);
        s.drain(2);
        assert_eq!(s.len(), 1);
        assert_eq!(
            s.buffer(),
            &[Complex::new(normalize_sc16_q11(5), normalize_sc16_q11(6))]
        );
    }
}
