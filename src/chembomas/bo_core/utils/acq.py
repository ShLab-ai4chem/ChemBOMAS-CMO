import torch
from torch import Tensor
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)
from botorch.exceptions import InputDataError, UnsupportedError
from .misc import correct_indices
def _split_batch_eval_acqf(
    acq_function: AcquisitionFunction, X: Tensor, max_batch_size: int
) -> Tensor:
    return torch.cat([acq_function(X_) for X_ in X.split(max_batch_size)])

def optimize_acqf_discrete_idx(
    acq_function: AcquisitionFunction,
    q: int,
    choices: Tensor,
    max_batch_size: int = 2048,
    unique: bool = True,
    return_acq_values: bool = False,
) -> tuple[Tensor, Tensor]:
    r"""Optimize over a discrete set of points using batch evaluation.

    For `q > 1` this function generates candidates by means of sequential
    conditioning (rather than joint optimization), since for all but the
    smalles number of choices the set `choices^q` of discrete points to
    evaluate quickly explodes.

    Args:
        acq_function: An AcquisitionFunction.
        q: The number of candidates.
        choices: A `num_choices x d` tensor of possible choices.
        max_batch_size: The maximum number of choices to evaluate in batch.
            A large limit can cause excessive memory usage if the model has
            a large training set.
        unique: If True return unique choices, o/w choices may be repeated
            (only relevant if `q > 1`).

    Returns:
        A two-element tuple containing

        - a `q x d`-dim tensor of generated candidates.
        - an associated acquisition value.
    """
    if isinstance(acq_function, OneShotAcquisitionFunction):
        raise UnsupportedError(
            "Discrete optimization is not supported for"
            "one-shot acquisition functions."
        )
    if choices.numel() == 0:
        raise InputDataError("`choices` must be non-emtpy.")
    choices_batched = choices.unsqueeze(-2)
    if q > 1:
        candidate_list, acq_value_list,best_idxes,mask = [], [], [], []
        base_X_pending = acq_function.X_pending
        for _ in range(q):
            with torch.no_grad():
                acq_values = _split_batch_eval_acqf(
                    acq_function=acq_function,
                    X=choices_batched,
                    max_batch_size=max_batch_size,
                )
        
            best_idx = torch.argmax(acq_values)
            mask_tensor = torch.zeros(choices.shape[0], dtype=torch.bool)
            mask_indices = torch.tensor([m.item() for m in mask], dtype=torch.long)
            mask_tensor[mask_indices] = True
            best_idxes.append(best_idx)
            candidate_list.append(choices_batched[best_idx])
            acq_value_list.append(acq_values[best_idx])
            # set pending points
            candidates = torch.cat(candidate_list, dim=-2)
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2)
                if base_X_pending is not None
                else candidates
            )
            # need to remove choice from choice set if enforcing uniqueness
            
            if unique:
                choices_batched = torch.cat(
                    [choices_batched[:best_idx], choices_batched[best_idx + 1 :]]
                )
                #best_idxes = best_idxes.remove(best_idx)
                mask.append(correct_indices(best_idx,mask_tensor))

        # Reset acq_func to previous X_pending state
        acq_function.set_X_pending(base_X_pending)
        # return candidates, torch.stack(acq_value_list)
        if return_acq_values:
            return torch.tensor(best_idxes).tolist(), acq_value_list
        else:
            return torch.tensor(best_idxes)

    with torch.no_grad():
        acq_values = _split_batch_eval_acqf(
            acq_function=acq_function, X=choices_batched, max_batch_size=max_batch_size
        )
    best_idx = torch.argmax(acq_values)
    # return choices[best_idx], acq_values[best_idx]
    if return_acq_values:
        return best_idx.clone().detach(), acq_values[best_idx]
    else:
        return best_idx.clone().detach()


def optimize_acqf_discrete_weighted_idx(
    acq_function: AcquisitionFunction,
    q: int,
    choices: Tensor,
    weights: Tensor,
    max_batch_size: int = 2048,
    unique: bool = True,
    
) -> tuple[Tensor, Tensor]:
    r"""Optimize over a discrete set of points using batch evaluation.

    For `q > 1` this function generates candidates by means of sequential
    conditioning (rather than joint optimization), since for all but the
    smalles number of choices the set `choices^q` of discrete points to
    evaluate quickly explodes.

    Args:
        acq_function: An AcquisitionFunction.
        q: The number of candidates.
        choices: A `num_choices x d` tensor of possible choices.
        max_batch_size: The maximum number of choices to evaluate in batch.
            A large limit can cause excessive memory usage if the model has
            a large training set.
        unique: If True return unique choices, o/w choices may be repeated
            (only relevant if `q > 1`).

    Returns:
        A two-element tuple containing

        - a `q x d`-dim tensor of generated candidates.
        - an associated acquisition value.
    """
    if isinstance(acq_function, OneShotAcquisitionFunction):
        raise UnsupportedError(
            "Discrete optimization is not supported for"
            "one-shot acquisition functions."
        )
    if choices.numel() == 0:
        raise InputDataError("`choices` must be non-emtpy.")
    choices_batched = choices.unsqueeze(-2)
    if q > 1:
        candidate_list, acq_value_list,best_idxes,mask = [], [], [], []
        base_X_pending = acq_function.X_pending
        for _ in range(q):
            with torch.no_grad():
                acq_values = _split_batch_eval_acqf(
                    acq_function=acq_function,
                    X=choices_batched,
                    max_batch_size=max_batch_size,
                )
                acq_values = acq_values * weights
            mask_tensor = torch.zeros(choices.shape[0], dtype=torch.bool)
            mask_indices = torch.tensor([m.item() for m in mask], dtype=torch.long)
            mask_tensor[mask_indices] = True
            best_idx = torch.argmax(acq_values)
            best_idxes.append(correct_indices(best_idx,mask_tensor))
            candidate_list.append(choices_batched[best_idx])
            acq_value_list.append(acq_values[best_idx])
            # set pending points
            candidates = torch.cat(candidate_list, dim=-2)
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2)
                if base_X_pending is not None
                else candidates
            )
            # need to remove choice from choice set if enforcing uniqueness
            
            if unique:
                choices_batched = torch.cat(
                    [choices_batched[:best_idx], choices_batched[best_idx + 1 :]]
                )
                #best_idxes = best_idxes.remove(best_idx)
                weights = torch.cat(
                    [weights[:best_idx], weights[best_idx + 1 :]]
                )
                mask.append(correct_indices(best_idx,mask_tensor))
            

        # Reset acq_func to previous X_pending state
        acq_function.set_X_pending(base_X_pending)
        # return candidates, torch.stack(acq_value_list)
        return torch.tensor(best_idxes)

    with torch.no_grad():
        acq_values = _split_batch_eval_acqf(
            acq_function=acq_function, X=choices_batched, max_batch_size=max_batch_size
        )
        acq_values = acq_values * weights
    best_idx = torch.argmax(acq_values)
    # return choices[best_idx], acq_values[best_idx]
    return best_idx.clone().detach()
