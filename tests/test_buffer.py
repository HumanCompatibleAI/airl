import numpy as np
import pytest

from imitation.data import types
from imitation.data.buffer import Buffer, ReplayBuffer


def _fill_chunk(start, chunk_len, sample_shape, dtype=np.float):
    fill_vals = np.arange(start, start + chunk_len, dtype=dtype)
    fill_vals = np.reshape(fill_vals, (-1,) + (1,) * len(sample_shape))
    chunk = np.tile(fill_vals, (1,) + sample_shape)
    return chunk


def _get_fill_from_chunk(chunk):
    chunk_len, *sample_shape = chunk.shape
    sample_size = max(1, np.prod(sample_shape))
    return chunk.flatten()[::sample_size]


def _check_bound(end, capacity, samples, offset=0):
    start = max(0, end - capacity)
    assert np.all(start + offset <= samples), "samples violate lower bound"
    assert np.all(samples <= end + offset), "samples violate upper bound"


@pytest.mark.parametrize("capacity", [10, 30, 60])
@pytest.mark.parametrize("chunk_len", [1, 2, 4, 9])
@pytest.mark.parametrize("sample_shape", [(), (1, 2), (5, 4, 4)])
def test_buffer(capacity, chunk_len, sample_shape):
    """Builds a Buffer with the provided `capacity` and insert `capacity * 3`
    samples into the buffer in chunks of shape `(chunk_len,) + sample_shape`.

    We always insert chunks with consecutive integers.

    * `len(buffer)` should increase until we reach capacity.
    * `buffer._idx` should loop between 0 and `capacity - 1`.
    * After every insertion, samples should be in expected range, verifying
      FIFO insertion.
    * Mutating the inserted chunk shouldn't mutate the buffer.
    """
    buf = Buffer(
        capacity,
        sample_shapes={"a": sample_shape, "b": sample_shape},
        dtypes={"a": float, "b": float},
    )

    to_insert = 3 * capacity
    for i in range(0, to_insert, chunk_len):
        assert buf.size() == min(i, capacity)
        assert buf._idx == i % capacity
        chunk_a = _fill_chunk(i, chunk_len, sample_shape)
        chunk_b = _fill_chunk(i + to_insert, chunk_len, sample_shape)
        buf.store({"a": chunk_a, "b": chunk_b})
        samples = buf.sample(100)
        assert set(samples.keys()) == {"a", "b"}, samples.keys()
        _check_bound(i + chunk_len, capacity, samples["a"])
        _check_bound(i + chunk_len + to_insert, capacity, samples["b"])
        assert np.all(samples["b"] - samples["a"] == to_insert)

        # Confirm that buffer is not mutable from inserted sample.
        chunk_a[:] = np.nan
        chunk_b[:] = np.nan
        assert not np.any(np.isnan(buf._arrays["a"]))
        assert not np.any(np.isnan(buf._arrays["b"]))


@pytest.mark.parametrize("capacity", [30, 60])
@pytest.mark.parametrize("chunk_len", [1, 4, 9])
@pytest.mark.parametrize("obs_shape", [(), (1, 2)])
@pytest.mark.parametrize("act_shape", [(), (5, 4, 4)])
@pytest.mark.parametrize("dtype", [int, np.float32])
def test_replay_buffer(capacity, chunk_len, obs_shape, act_shape, dtype):
    """Builds a ReplayBuffer with the provided `capacity` and inserts.

    `capacity * 3` observation-action-observation samples into the buffer in
    chunks of length `chunk_len`.

    All chunks are of the appropriate observation or action shape, and contain
    the value fill_val.

    `len(buffer)` should increase until we reach capacity.
    `buffer._idx` should loop between 0 and `capacity - 1`.
    After every insertion, samples should only contain 66.6.
    """
    buf = ReplayBuffer(
        capacity,
        obs_shape=obs_shape,
        act_shape=act_shape,
        obs_dtype=dtype,
        act_dtype=dtype,
    )

    for i in range(0, capacity * 3, chunk_len):
        assert buf.size() == min(i, capacity)
        assert buf._buffer._idx == i % capacity

        dones = np.arange(i, i + chunk_len, dtype=np.int32) % 2
        dones = dones.astype(bool)
        infos = _fill_chunk(9 * capacity + i, chunk_len, (), dtype=dtype)
        infos = np.array([{"a": val} for val in infos])
        batch = types.Transitions(
            obs=_fill_chunk(i, chunk_len, obs_shape, dtype=dtype),
            next_obs=_fill_chunk(3 * capacity + i, chunk_len, obs_shape, dtype=dtype),
            acts=_fill_chunk(6 * capacity + i, chunk_len, act_shape, dtype=dtype),
            dones=dones,
            infos=infos,
        )
        buf.store(batch)

        # Are samples right shape?
        sample = buf.sample(100)
        info_vals = np.array([info["a"] for info in sample.infos])

        assert sample.obs.shape == sample.next_obs.shape == (100,) + obs_shape
        assert sample.acts.shape == (100,) + act_shape
        assert sample.dones.shape == (100,)
        assert info_vals.shape == (100,)

        # Are samples right data type?
        assert sample.obs.dtype == dtype
        assert sample.acts.dtype == dtype
        assert sample.next_obs.dtype == dtype
        assert info_vals.dtype == dtype
        assert sample.dones.dtype == bool
        assert sample.infos.dtype == np.object

        # Are samples in range?
        _check_bound(i + chunk_len, capacity, sample.obs)
        _check_bound(i + chunk_len, capacity, sample.next_obs, 3 * capacity)
        _check_bound(i + chunk_len, capacity, sample.acts, 6 * capacity)
        _check_bound(i + chunk_len, capacity, info_vals, 9 * capacity)

        # Are samples in-order?
        obs_fill = _get_fill_from_chunk(sample.obs)
        next_obs_fill = _get_fill_from_chunk(sample.next_obs)
        act_fill = _get_fill_from_chunk(sample.acts)
        info_vals_fill = _get_fill_from_chunk(info_vals)

        assert np.all(next_obs_fill - obs_fill == 3 * capacity), "out of order"
        assert np.all(act_fill - next_obs_fill == 3 * capacity), "out of order"
        assert np.all(info_vals_fill - act_fill == 3 * capacity), "out of order"
        # Can't do much other than parity check for boolean values.
        # `samples.done` has the same parity as `obs_fill` by construction.
        assert np.all(obs_fill % 2 == sample.dones), "out of order"


@pytest.mark.parametrize("sample_shape", [(), (1,), (5, 2)])
def test_buffer_store_errors(sample_shape):
    capacity = 11
    dtype = "float32"

    def buf():
        return Buffer(capacity, {"k": sample_shape}, {"k": dtype})

    # Unexpected keys
    b = buf()
    with pytest.raises(ValueError):
        b.store({})
    chunk = np.ones((1,) + sample_shape)
    with pytest.raises(ValueError):
        b.store({"y": chunk})
    with pytest.raises(ValueError):
        b.store({"k": chunk, "y": chunk})

    # `data` is empty.
    b = buf()
    with pytest.raises(ValueError):
        b.store({"k": np.empty((0,) + sample_shape, dtype=dtype)})

    # `data` has too many samples.
    b = buf()
    with pytest.raises(ValueError):
        b.store({"k": np.empty((capacity + 1,) + sample_shape, dtype=dtype)})

    # `data` has the wrong sample shape.
    b = buf()
    with pytest.raises(ValueError):
        b.store({"k": np.empty((1, 3, 3, 3, 3), dtype=dtype)})


def test_buffer_sample_errors():
    b = Buffer(10, {"k": (2, 1)}, dtypes={"k": bool})
    with pytest.raises(ValueError):
        b.sample(5)


def test_buffer_init_errors():
    with pytest.raises(KeyError, match=r"sample_shape and dtypes.*"):
        Buffer(10, dict(a=(2, 1), b=(3,)), dtypes=dict(a="float32", c=bool))


def test_replay_buffer_init_errors():
    with pytest.raises(ValueError, match=r"Specified.* and environment"):
        ReplayBuffer(15, venv="MockEnv", obs_shape=(10, 10))
    with pytest.raises(ValueError, match=r"Shape or dtype missing.*"):
        ReplayBuffer(15, obs_shape=(10, 10), act_shape=(15,), obs_dtype=bool)
    with pytest.raises(ValueError, match=r"Shape or dtype missing.*"):
        ReplayBuffer(15, obs_shape=(10, 10), obs_dtype=bool, act_dtype=bool)


def test_buffer_from_data():
    data = np.ndarray([50, 30], dtype=bool)
    buf = Buffer.from_data({"k": data})
    assert buf._arrays["k"] is not data
    assert data.dtype == buf._arrays["k"].dtype
    assert np.array_equal(buf._arrays["k"], data)


def test_replay_buffer_from_data():
    obs = np.array([5, 2], dtype=int)
    acts = np.ones((2, 6), dtype=float)
    next_obs = np.array([7, 8], dtype=int)
    dones = np.array([True, False])
    infos = np.array([{}, {"a": "sdf"}])

    def _check_buf(buf):
        assert np.array_equal(buf._buffer._arrays["obs"], obs)
        assert np.array_equal(buf._buffer._arrays["next_obs"], next_obs)
        assert np.array_equal(buf._buffer._arrays["acts"], acts)
        assert np.array_equal(buf._buffer._arrays["infos"], infos)

    buf_std = ReplayBuffer.from_data(
        types.Transitions(
            obs=obs, acts=acts, next_obs=next_obs, dones=dones, infos=infos
        )
    )
    _check_buf(buf_std)

    rews = np.array([0.5, 1.0], dtype=float)
    buf_rew = ReplayBuffer.from_data(
        types.TransitionsWithRew(
            obs=obs,
            acts=acts,
            next_obs=next_obs,
            rews=rews,
            dones=dones,
            infos=infos,
        )
    )
    _check_buf(buf_rew)
