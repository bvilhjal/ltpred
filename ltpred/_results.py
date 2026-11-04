"""Optional table export for liability scores; pandas is never an import dependency."""


class _TableExport:
    def to_frame(self):
        """Return the copied `to_dict()` columns as a pandas DataFrame.

        Pandas is an optional dependency; repeated ids remain separate rows.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("to_frame() requires pandas; install pandas or use to_dict()") from None
        return pd.DataFrame(self.to_dict())
