import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_generator(seed: int, offset: int = 0) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(offset))
    return generator


def worker_init_fn(wid):
    seed_sequence = np.random.SeedSequence(
        [torch.initial_seed(), wid]
    )

    to_seed = spawn_get(seed_sequence, 2, dtype=int)
    torch.random.manual_seed(to_seed)

    np_seed = spawn_get(seed_sequence, 2, dtype=np.ndarray)
    np.random.seed(np_seed)

    py_seed = spawn_get(seed_sequence, 2, dtype=int)
    random.seed(py_seed)


def spawn_get(seedseq, n_entropy, dtype):
    child = seedseq.spawn(1)[0]
    state = child.generate_state(n_entropy, dtype=np.uint32)

    if dtype == np.ndarray:
        return state
    elif dtype == int:
        state_as_int = 0
        for shift, s in enumerate(state):
            state_as_int += s << (32 * shift)
        return state_as_int
    else:
        raise ValueError(f'not a valid dtype "{dtype}"')
