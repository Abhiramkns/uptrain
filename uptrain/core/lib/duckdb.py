import numpy as np
import pyarrow as pa


def array_np_to_arrow(arr: np.ndarray):
    if arr.ndim == 1:
        return pa.array(arr)
    else:
        assert arr.ndim == 2
        dim1, dim2 = arr.shape
        return pa.ListArray.from_arrays(
            np.arange(0, (dim1 + 1) * dim2, dim2), arr.ravel()
        )
    

def array_arrow_to_np(arr: pa.ListArray):
    if not arr:
        # Empty arr
        return arr.to_numpy()
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    if not pa.types.is_list(arr.type):
        return arr.to_numpy()
    else:
        # assumes a 2D array
        dim1 = len(arr)
        return np.asarray(arr.values.to_numpy()).reshape(dim1, -1)


# This function does not work
# def add_id_values(conn, tbl_name, ids: np.ndarray, values: np.ndarray):
#     tbl = pa.Table.from_pydict(
#         {
#             "id": array_np_to_arrow(ids),
#             "value": array_np_to_arrow(values),
#         }
#     )
#     try:
#         conn.execute(f"""INSERT INTO {tbl_name} 
#             SELECT id, value FROM tbl
#             WHERE id NOT IN (SELECT id FROM {tbl_name})""")
#     except:
#         import pdb; pdb.set_trace()
    

def upsert_id_values(conn, tbl_name, ids: np.ndarray, values: np.ndarray):
    tbl = pa.Table.from_pydict(
        {
            "id": array_np_to_arrow(ids),
            "value": array_np_to_arrow(values),
        }
    )
    # upserts don't work for List data types, so we do it in steps
    # conn.execute("INSERT OR REPLACE INTO intermediates SELECT id, value FROM tbl")
    conn.execute(f"DELETE FROM {tbl_name} WHERE id IN (SELECT id FROM tbl)")
    conn.execute(f"INSERT INTO {tbl_name} SELECT id, value FROM tbl")


def fetch_values_for_id(conn, tbl_name, id_list):
    id_tbl = pa.Table.from_pydict({"id": id_list})
    conn.execute(f"SELECT id, value FROM {tbl_name} where id IN (SELECT id FROM id_tbl)")
    return conn.fetchnumpy()
    # ar_tbl = conn.fetch_arrow_table()
    # return array_arrow_to_np(ar_tbl['value'])
