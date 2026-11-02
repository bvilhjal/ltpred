"""Optional table export for liability scores; pandas is never an import dependency."""


class _TableExport:
    def to_frame(self):
        """Return `to_dict()` as a pandas DataFrame with a default row index
        (with id columns where the result carries ids; `QuadratureResult` has none).

        Requires pandas. Repeated ids remain separate rows; use the id column
        for a checked join. Arrays are copied so editing the table cannot alter
        the numerical result.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("to_frame() requires pandas; install pandas or use to_dict()") from None
        return pd.DataFrame(self.to_dict())
