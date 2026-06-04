use thiserror::Error;

#[derive(Debug, Error)]
pub enum NcError {
    #[error("HDF5 error: {0}")]
    H5(#[from] oxih5::OxiH5Error),
    #[error("dimension '{0}' is missing the _Netcdf4Dimid attribute")]
    MissingDimId(String),
    #[error("duplicate _Netcdf4Dimid {0}")]
    DuplicateDimId(u32),
    #[error(
        "variable '{var}' axis {axis}: dimension reference did not resolve to a dimension scale"
    )]
    UnresolvedDimRef { var: String, axis: usize },
    #[error("variable '{var}' axis {axis}: unknown dimension id {dim_id}")]
    UnknownDimId {
        var: String,
        axis: usize,
        dim_id: u32,
    },
    #[error("variable '{var}': DIMENSION_LIST has {found} entries but dataset has rank {rank}")]
    DimensionListArity {
        var: String,
        found: usize,
        rank: usize,
    },
    #[error("variable '{var}' axis {axis}: length {var_len} != dimension length {dim_len}")]
    AxisLengthMismatch {
        var: String,
        axis: usize,
        var_len: u64,
        dim_len: u64,
    },
    #[error("attribute '{attr}' on '{owner}' could not be decoded: {reason}")]
    BadConventionAttribute {
        owner: String,
        attr: String,
        reason: String,
    },
    #[error("variable not found: {0}")]
    VariableNotFound(String),
    #[error("not supported in this release: {0}")]
    Unsupported(String),
}
