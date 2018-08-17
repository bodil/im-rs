use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::yield_now;

/// The simplest possible write-only spin lock.
#[derive(Clone)]
pub struct TinyLock {
    lock: Arc<AtomicBool>,
}

impl TinyLock {
    pub fn new() -> Self {
        TinyLock {
            lock: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn lock(&mut self) -> LockHandle {
        while self.lock.fetch_or(true, Ordering::SeqCst) != false {
            yield_now()
        }
        LockHandle {
            lock: self.lock.clone(),
        }
    }
}

pub struct LockHandle {
    lock: Arc<AtomicBool>,
}

impl Drop for LockHandle {
    fn drop(&mut self) {
        self.lock.store(false, Ordering::SeqCst);
    }
}
