//! Workflow state-machine adapter backed by reusable hybrid state references.

use serde::{Deserialize, Serialize};

use crate::state_id::HybridStateId;

/// A durable checkpoint of a workflow's progress.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkflowSnapshot {
    /// Stable identifier for the workflow instance.
    pub workflow_id: String,
    /// Content-addressed state references, in workflow step order.
    pub step_states: Vec<HybridStateId>,
}

impl WorkflowSnapshot {
    /// Capture the current state of a workflow.
    pub fn checkpoint<I, S>(workflow_id: I, step_states: S) -> Self
    where
        I: Into<String>,
        S: IntoIterator<Item = HybridStateId>,
    {
        Self {
            workflow_id: workflow_id.into(),
            step_states: step_states.into_iter().collect(),
        }
    }

    /// Return the state references needed to resume this workflow.
    pub fn resume(&self) -> Vec<HybridStateId> {
        self.step_states.clone()
    }
}

/// In-memory adapter representing the current workflow state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkflowStateMachine {
    pub workflow_id: String,
    pub step_states: Vec<HybridStateId>,
}

impl WorkflowStateMachine {
    pub fn new(workflow_id: impl Into<String>) -> Self {
        Self {
            workflow_id: workflow_id.into(),
            step_states: Vec::new(),
        }
    }

    /// Append the state reference produced by the next workflow step.
    pub fn push_state(&mut self, state: HybridStateId) {
        self.step_states.push(state);
    }

    pub fn checkpoint(&self) -> WorkflowSnapshot {
        WorkflowSnapshot::checkpoint(self.workflow_id.clone(), self.step_states.clone())
    }

    /// Restore this adapter from a checkpoint and return the restored state.
    pub fn resume(&mut self, snapshot: &WorkflowSnapshot) -> Vec<HybridStateId> {
        self.workflow_id = snapshot.workflow_id.clone();
        self.step_states = snapshot.resume();
        self.step_states.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(value: &str) -> HybridStateId {
        HybridStateId(value.to_owned())
    }

    #[test]
    fn checkpoint_captures_workflow_and_step_order() {
        let snapshot = WorkflowSnapshot::checkpoint("wf-1", vec![state("s1"), state("s2")]);
        assert_eq!(snapshot.workflow_id, "wf-1");
        assert_eq!(snapshot.step_states, vec![state("s1"), state("s2")]);
    }

    #[test]
    fn resume_reuses_checkpointed_states() {
        let snapshot = WorkflowSnapshot::checkpoint("wf-1", vec![state("s1"), state("s2")]);
        assert_eq!(snapshot.resume(), vec![state("s1"), state("s2")]);
    }

    #[test]
    fn adapter_round_trips_checkpoint() {
        let mut machine = WorkflowStateMachine::new("wf-1");
        machine.push_state(state("s1"));
        machine.push_state(state("s2"));
        let snapshot = machine.checkpoint();

        let mut restored = WorkflowStateMachine::new("other");
        assert_eq!(restored.resume(&snapshot), vec![state("s1"), state("s2")]);
        assert_eq!(restored, machine);
    }
}
