//! CF-1.x convention parsing helpers (B8).
//!
//! These are pure functions — they operate only on string slices and return
//! owned `Vec<String>` or `Option<String>`.  No HDF5 file I/O is performed
//! here.  The `NcGroup` methods `coordinates_of`, `bounds_of`, and
//! `grid_mapping_of` delegate to these helpers.

/// Parse a CF-convention space-separated list of variable names.
///
/// CF-1.7 §6.2 uses a whitespace-separated list for the `coordinates`
/// attribute, where each token may optionally use the `"group:varname"` form
/// to reference a variable in a different group.  This function returns each
/// token as-is (preserving the `group:var` colon notation if present).
///
/// Returns an empty `Vec` if `s` is all whitespace.
///
/// # Examples
///
/// ```
/// # use oxinetcdf::cf::parse_cf_name_list;
/// assert_eq!(
///     parse_cf_name_list("lat lon"),
///     vec!["lat".to_string(), "lon".to_string()]
/// );
/// assert_eq!(
///     parse_cf_name_list("grp:lat grp:lon"),
///     vec!["grp:lat".to_string(), "grp:lon".to_string()]
/// );
/// ```
pub fn parse_cf_name_list(s: &str) -> Vec<String> {
    s.split_whitespace().map(|tok| tok.to_string()).collect()
}

/// Extract the group prefix from a CF `"group:varname"` token.
///
/// Returns `Some("group")` for `"group:var"` and `None` for bare `"var"`.
///
/// Only the first colon is treated as the separator; colons within a variable
/// name (which the CF conventions do not support) are retained as-is.
pub fn cf_group_prefix(token: &str) -> Option<&str> {
    let idx = token.find(':')?;
    Some(&token[..idx])
}

/// Extract the variable-name part of a CF `"group:varname"` token.
///
/// Returns `"var"` for `"group:var"` and the whole token for `"var"`.
pub fn cf_var_name(token: &str) -> &str {
    match token.find(':') {
        Some(idx) => &token[idx + 1..],
        None => token,
    }
}

// ---------------------------------------------------------------------------
// Unit tests (B8)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cf_name_list_simple() {
        let result = parse_cf_name_list("lat lon time");
        assert_eq!(
            result,
            vec!["lat".to_string(), "lon".to_string(), "time".to_string()]
        );
    }

    #[test]
    fn test_parse_cf_name_list_single() {
        let result = parse_cf_name_list("lat");
        assert_eq!(result, vec!["lat".to_string()]);
    }

    #[test]
    fn test_parse_cf_name_list_empty() {
        assert!(parse_cf_name_list("").is_empty());
        assert!(parse_cf_name_list("   ").is_empty());
    }

    #[test]
    fn test_parse_cf_name_list_colon_form() {
        // CF-1.7 group:var form — tokens are returned verbatim.
        let result = parse_cf_name_list("group1:lat group1:lon");
        assert_eq!(
            result,
            vec!["group1:lat".to_string(), "group1:lon".to_string()]
        );
    }

    #[test]
    fn test_parse_cf_name_list_extra_whitespace() {
        let result = parse_cf_name_list("  lat   lon  ");
        assert_eq!(result, vec!["lat".to_string(), "lon".to_string()]);
    }

    #[test]
    fn test_cf_group_prefix_with_colon() {
        assert_eq!(cf_group_prefix("group1:lat"), Some("group1"));
        assert_eq!(cf_group_prefix("g:v"), Some("g"));
    }

    #[test]
    fn test_cf_group_prefix_bare_name() {
        assert_eq!(cf_group_prefix("lat"), None);
        assert_eq!(cf_group_prefix(""), None);
    }

    #[test]
    fn test_cf_var_name_with_colon() {
        assert_eq!(cf_var_name("group1:lat"), "lat");
        assert_eq!(cf_var_name("g:v"), "v");
    }

    #[test]
    fn test_cf_var_name_bare() {
        assert_eq!(cf_var_name("lat"), "lat");
        assert_eq!(cf_var_name(""), "");
    }
}
