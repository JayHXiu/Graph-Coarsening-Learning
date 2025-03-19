from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from torch import Tensor
from torch_geometric.data.database import SQLiteDatabase as SQLiteDatabase_ori


class SQLiteDatabase(SQLiteDatabase_ori):
    def multi_get(
        self,
        indices: Union[Iterable[int], Tensor, slice, range],
        batch_size: Optional[int] = None,
    ) -> List[Any]:

        if isinstance(indices, slice):
            indices = self.slice_to_range(indices)
        elif isinstance(indices, Tensor):
            indices = indices.tolist()

        # Nah, we pass the ids altogether
        indices_str = ', '.join([str(i) for i in indices])
        query = (f'SELECT {self._joined_col_names} FROM {self.name} '
            f'WHERE id in ( {indices_str} ) ORDER BY id')
        self.cursor.execute(query)
        data_list = self.cursor.fetchall()
        return [self._deserialize(data) for data in data_list]
