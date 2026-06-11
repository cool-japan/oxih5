/// HDF5 attribute names that are NetCDF-4 internal conventions.
/// These are filtered OUT of the user-facing NcAttribute list.
/// (But _FillValue, _NCProperties, units, etc. are kept — they are user-meaningful.)
pub(crate) const RESERVED_ATTRS: &[&str] = &[
    "CLASS",
    "NAME",
    "DIMENSION_LIST",
    "REFERENCE_LIST",
    "_Netcdf4Dimid",
    "_Netcdf4Coordinates",
    "DIMENSION_LABELS",
];

/// The sentinel prefix written by netCDF4 for a pure (no-coordinate-variable) dimension.
/// Full form: "This is a netCDF dimension but not a netCDF variable.<len>"
const PURE_DIM_SENTINEL: &str = "This is a netCDF dimension but not a netCDF variable.";

/// Try to parse a NAME attribute value as a pure-dimension sentinel.
/// Returns `Some(len)` if it is the sentinel (the length encoded in the suffix),
/// or `None` if it is an ordinary coordinate-variable NAME.
pub(crate) fn parse_pure_dim_sentinel(name: &str) -> Option<u64> {
    let suffix = name.strip_prefix(PURE_DIM_SENTINEL)?;
    suffix.trim().parse::<u64>().ok()
}

/// Returns true if `name` is a reserved NetCDF-4 convention attribute that
/// should be hidden from the user-facing attribute list.
pub(crate) fn is_reserved_attr(name: &str) -> bool {
    RESERVED_ATTRS.contains(&name)
}

/// Generate a phony dimension name for an axis that has no DIMENSION_LIST entry.
/// Convention matches netCDF library: "phony_dim_N" where N counts from 0 across
/// all phony dims allocated within a group.
pub(crate) fn phony_dim_name(index: u32) -> String {
    format!("phony_dim_{index}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pure_dim_sentinel_parses_len() {
        let s = "This is a netCDF dimension but not a netCDF variable.42";
        assert_eq!(parse_pure_dim_sentinel(s), Some(42));
    }

    #[test]
    fn test_pure_dim_sentinel_with_spaces() {
        let s = "This is a netCDF dimension but not a netCDF variable. 10";
        assert_eq!(parse_pure_dim_sentinel(s), Some(10));
    }

    #[test]
    fn test_non_sentinel_returns_none() {
        assert_eq!(parse_pure_dim_sentinel("time"), None);
        assert_eq!(parse_pure_dim_sentinel("lat"), None);
        assert_eq!(
            parse_pure_dim_sentinel("This is a netCDF dimension but not a netCDF variable.abc"),
            None
        );
    }

    #[test]
    fn test_reserved_attr_filter() {
        assert!(is_reserved_attr("CLASS"));
        assert!(is_reserved_attr("DIMENSION_LIST"));
        assert!(is_reserved_attr("_Netcdf4Dimid"));
        // user-meaningful attrs must NOT be filtered
        assert!(!is_reserved_attr("units"));
        assert!(!is_reserved_attr("_FillValue"));
        assert!(!is_reserved_attr("_NCProperties"));
        assert!(!is_reserved_attr("long_name"));
    }

    #[test]
    fn test_phony_dim_naming() {
        assert_eq!(phony_dim_name(0), "phony_dim_0");
        assert_eq!(phony_dim_name(3), "phony_dim_3");
    }
}
