"""Optional table export for liability scores; pandas is never an import dependency."""


class _TableExport:
    def to_frame(self):
        """Return a pandas DataFrame with explicit id columns and a row index.

        Requires pandas. Repeated ids remain separate rows; use the id column
        for a checked join. Arrays are copied so editing the table cannot alter
        the numerical result.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("to_frame() requires pandas; install pandas or use to_dict()") from None
        return pd.DataFrame(self.to_dict())
