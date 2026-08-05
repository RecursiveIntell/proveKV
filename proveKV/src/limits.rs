use serde::{Deserialize, Serialize};

use crate::error::{ProveKvError, Result};

/// Predeclared runtime bounds; all limits are hard ceilings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Raw manifest payload ceiling in bytes.
    pub max_manifest_bytes: Option<u64>,
    /// Maximum number of pages across all live manifests.
    pub max_page_count: Option<u64>,
    /// Maximum single page size in bytes.
    pub max_page_bytes: Option<u64>,
    /// Maximum allowed component count.
    pub max_component_count: Option<u64>,
    /// Maximum allowed layer count.
    pub max_layer_count: Option<u64>,
    /// Maximum allowed rank value.
    pub max_rank: Option<u64>,
    /// Maximum allowed dimension value.
    pub max_dimension: Option<u64>,
    /// Total live state payload cap.
    pub max_total_state_bytes: Option<u64>,
    /// Maximum branch depth per state.
    pub max_branch_depth: Option<u64>,
    /// Live states per principal cap.
    pub max_live_states_per_principal: Option<u64>,
    /// Concurrent in-flight calls cap.
    pub max_concurrent_requests: Option<u64>,
    /// Decode/materialize micro-budget cap.
    pub max_decode_budget: Option<u64>,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_manifest_bytes: Some(32 * 1024 * 1024),
            max_page_count: Some(1024),
            max_page_bytes: Some(64 * 1024 * 1024),
            max_component_count: Some(2048),
            max_layer_count: Some(256),
            max_rank: Some(8192),
            max_dimension: Some(8192),
            max_total_state_bytes: Some(32 * 1024 * 1024 * 1024),
            max_branch_depth: Some(1024),
            max_live_states_per_principal: Some(256),
            max_concurrent_requests: Some(128),
            max_decode_budget: Some(4 * 60),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub manifest_bytes: u64,
    pub page_count: u64,
    pub page_bytes_max: u64,
    pub component_count: u64,
    pub layer_count: u64,
    pub rank: u64,
    pub dimension: u64,
    pub total_state_bytes: u64,
    pub branch_depth: u64,
    pub live_states_per_principal: u64,
    pub concurrent_requests: u64,
    pub decode_budget: u64,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            manifest_bytes: 0,
            page_count: 0,
            page_bytes_max: 0,
            component_count: 0,
            layer_count: 0,
            rank: 0,
            dimension: 0,
            total_state_bytes: 0,
            branch_depth: 0,
            live_states_per_principal: 0,
            concurrent_requests: 0,
            decode_budget: 0,
        }
    }
}

impl ResourceLimits {
    /// Validate all observed usage against hard limits.
    pub fn validate_usage(&self, usage: &ResourceUsage) -> Result<()> {
        self.check(
            "manifest_bytes",
            usage.manifest_bytes,
            self.max_manifest_bytes,
        )?;
        self.check("page_count", usage.page_count, self.max_page_count)?;
        self.check("page_bytes_max", usage.page_bytes_max, self.max_page_bytes)?;
        self.check(
            "component_count",
            usage.component_count,
            self.max_component_count,
        )?;
        self.check("layer_count", usage.layer_count, self.max_layer_count)?;
        self.check("rank", usage.rank, self.max_rank)?;
        self.check("dimension", usage.dimension, self.max_dimension)?;
        self.check(
            "total_state_bytes",
            usage.total_state_bytes,
            self.max_total_state_bytes,
        )?;
        self.check("branch_depth", usage.branch_depth, self.max_branch_depth)?;
        self.check(
            "live_states_per_principal",
            usage.live_states_per_principal,
            self.max_live_states_per_principal,
        )?;
        self.check(
            "concurrent_requests",
            usage.concurrent_requests,
            self.max_concurrent_requests,
        )?;
        self.check("decode_budget", usage.decode_budget, self.max_decode_budget)?;
        Ok(())
    }

    fn check(&self, what: &str, value: u64, cap: Option<u64>) -> Result<()> {
        if let Some(limit) = cap {
            if value > limit {
                return Err(ProveKvError::ResourceLimitExceeded(format!(
                    "{what} {} exceeds limit {limit}",
                    value
                )));
            }
        }
        Ok(())
    }

    /// Checked element count x stride, used for dimension-based allocation math.
    pub fn checked_area(&self, dims: &[u64]) -> Result<u64> {
        let mut total: u64 = 1;
        for d in dims {
            total = checked_mul(total, *d)?;
        }
        if let Some(max_dim) = self.max_dimension {
            if dims.iter().any(|d| *d > max_dim) {
                return Err(ProveKvError::ResourceLimitExceeded(format!(
                    "dimension {dims:?} exceeds configured max_dimension={max_dim}"
                )));
            }
        }
        Ok(total)
    }
}

fn checked_mul(lhs: u64, rhs: u64) -> Result<u64> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| ProveKvError::ResourceLimitExceeded("resource arithmetic overflow".into()))
}

#[cfg(test)]
mod tests {
    use super::{ResourceLimits, ResourceUsage};

    #[test]
    fn limits_pass_for_valid_usage() {
        let limits = ResourceLimits::default();
        let usage = ResourceUsage::default();
        limits.validate_usage(&usage).unwrap();
    }

    #[test]
    fn limits_rejects_overflow_or_excess() {
        let limits = ResourceLimits {
            max_dimension: Some(16),
            max_rank: Some(4),
            ..ResourceLimits::default()
        };

        let usage = ResourceUsage {
            dimension: 17,
            rank: 1,
            ..ResourceUsage::default()
        };
        assert!(limits.validate_usage(&usage).is_err());

        let overflow_dims = vec![u64::MAX, 2];
        assert!(limits
            .checked_area(&overflow_dims)
            .is_err_and(|e| format!("{e}").contains("overflow")));
    }
}
