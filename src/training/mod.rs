// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Training pipelines for the PC Actor-Critic agent.
//!
//! - [`trainer`] — Episode-based training with curriculum learning.
//! - [`continuous`] — Continuous training with surprise-based immediate updates.
//! - [`fitness`] — GA-compatible fitness scoring for trained agents.
//! - [`champion`] — Multi-session champion search with snapshot persistence.

pub mod champion;
pub mod continuous;
pub mod experiment;
pub mod fitness;
pub mod trainer;
