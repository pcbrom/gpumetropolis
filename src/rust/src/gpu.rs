//! Compute backend selection.

/// Compute backend. `Cpu` is the native Rust path; `Cuda` is the NVIDIA-native
/// path; `Vulkan` is the vendor-agnostic path through wgpu. The GPU backends
/// are compiled in only when the matching Cargo feature is enabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Cuda,
    Vulkan,
}

impl Backend {
    /// Parse a backend name; unknown names fall back to the CPU.
    pub fn from_name(name: &str) -> Backend {
        match name.to_ascii_lowercase().as_str() {
            "cuda" => Backend::Cuda,
            "vulkan" | "wgpu" => Backend::Vulkan,
            _ => Backend::Cpu,
        }
    }
}
