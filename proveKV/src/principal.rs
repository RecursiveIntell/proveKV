use serde::{Deserialize, Serialize};

use crate::error::{ProveKvError, Result};

/// Identity of a principal allowed to hold a lease.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Principal {
    /// Principal identifier (subject). Stable and non-empty.
    pub id: String,
    /// Logical namespace for isolation and principal-salted storage roots.
    pub namespace: String,
}

/// Execution/run scope used to prevent lease replay across attempts.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExecutionScope {
    /// Run identifier assigned by orchestrator/runner.
    pub run_id: String,
    /// Node identifier where execution is allowed.
    pub node_id: String,
    /// Attempt index inside the run.
    pub attempt: u32,
}

impl Principal {
    pub fn new<P: Into<String>, N: Into<String>>(id: P, namespace: N) -> Result<Self> {
        let principal = Self {
            id: id.into(),
            namespace: namespace.into(),
        };
        principal.validate()?;
        Ok(principal)
    }

    pub fn validate(&self) -> Result<()> {
        validate_non_empty_no_control(&self.id, "principal id")?;
        validate_non_empty_no_control(&self.namespace, "namespace")?;
        if self.id.len() > 128 {
            return Err(ProveKvError::InvalidPrincipal(
                "principal id exceeds max length 128".into(),
            ));
        }
        if self.namespace.len() > 256 {
            return Err(ProveKvError::InvalidPrincipal(
                "namespace exceeds max length 256".into(),
            ));
        }
        if !self.id.is_ascii() || !self.namespace.is_ascii() {
            return Err(ProveKvError::InvalidPrincipal(
                "principal id and namespace must be ASCII".into(),
            ));
        }
        Ok(())
    }

    /// Stable namespace prefix used for per-principal storage isolation.
    pub fn storage_prefix(&self) -> String {
        format!("principal:{}:{}", self.namespace, self.id)
    }
}

impl ExecutionScope {
    pub fn new<R: Into<String>, N: Into<String>>(
        run_id: R,
        node_id: N,
        attempt: u32,
    ) -> Result<Self> {
        let scope = Self {
            run_id: run_id.into(),
            node_id: node_id.into(),
            attempt,
        };
        scope.validate()?;
        Ok(scope)
    }

    pub fn validate(&self) -> Result<()> {
        validate_non_empty_no_control(&self.run_id, "run id")?;
        validate_non_empty_no_control(&self.node_id, "node id")?;
        Ok(())
    }

    pub fn key(&self) -> String {
        format!("{}:{}:{}", self.run_id, self.node_id, self.attempt)
    }
}

fn validate_non_empty_no_control(value: &str, label: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(ProveKvError::InvalidPrincipal(format!("{label} is empty")));
    }
    if value.chars().any(|c| c.is_control()) {
        return Err(ProveKvError::InvalidPrincipal(format!(
            "{label} contains control chars"
        )));
    }
    if value.len() > 256 {
        return Err(ProveKvError::InvalidPrincipal(format!(
            "{label} exceeds max length 256"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn principal_validation_and_prefix() {
        let principal = Principal::new("agent-a", "tenant-1").unwrap();
        assert_eq!(principal.storage_prefix(), "principal:tenant-1:agent-a");
    }

    #[test]
    fn execution_scope_key() {
        let scope = ExecutionScope::new("run-1", "node-1", 3).unwrap();
        assert_eq!(scope.key(), "run-1:node-1:3");
    }

    #[test]
    fn reject_control_chars() {
        assert!(Principal::new("agent\n", "ns").is_err());
        assert!(Principal::new("agent", "n\x00s").is_err());
        assert!(ExecutionScope::new("run", "no\nde", 0).is_err());
    }
}
