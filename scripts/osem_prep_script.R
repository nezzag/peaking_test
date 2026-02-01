#' Prepare local multi-country (+ROW) dataset + OSEM spec + dictionary
#'
#' @param energy A tibble with columns: region, region_name, year, prim_ener (and others ok)
#' @param gdp    A tibble with columns: region, region_name, year, gdp (and others ok)
#' @param co2    A tibble with columns: region, region_name, year, co2 (and others ok)
#' @param focus_countries Character vector of ISO3 country codes to model separately
#'                        e.g. c("DEU","USA","CHN")
#' @param include_row Logical; if TRUE create ROW = sum(all other regions)
#' @param year_to_date Logical; if TRUE convert numeric year to Date as YYYY-01-01
#'                     (recommended for {osem}; it expects a Date 'time' column)
#' @param log_safe_eps Small positive number used only if you later choose log transforms and
#'                     want to avoid exact zeros (not applied by default).
#'
#' @return A list with:
#'   - local_data: long tibble with columns time, na_item, values
#'   - specification: osem spec table (type/dependent/independent)
#'   - dictionary: minimal dictionary marking all vars as local/identity
#'   - wide_checks: optional wide tibbles useful for debugging (country + ROW series)
prepare_osem_co2_block <- function(
    energy,
    gdp,
    co2,
    focus_countries = c("DEU","USA","CHN"),
    include_row = TRUE,
    year_to_date = TRUE,
    log_safe_eps = 0
) {
  stopifnot(is.data.frame(energy), is.data.frame(gdp), is.data.frame(co2))
  stopifnot(is.character(focus_countries), length(focus_countries) >= 1)

  # ---- small helpers ----
  req_cols <- function(df, cols, df_name) {
    missing <- setdiff(cols, names(df))
    if (length(missing) > 0) {
      stop(sprintf(
        "%s is missing required columns: %s",
        df_name, paste(missing, collapse = ", ")
      ), call. = FALSE)
    }
  }

  make_time <- function(y) {
    if (!year_to_date) return(y)
    # turn 1965 -> "1965-01-01" Date
    as.Date(sprintf("%04d-01-01", as.integer(y)))
  }

  # ---- validate columns ----
  req_cols(energy, c("region","year","prim_ener"), "energy")
  req_cols(gdp,    c("region","year","gdp"),      "gdp")
  req_cols(co2,    c("region","year","co2"),      "co2")

  # ---- keep only needed columns, ensure unique region-year ----
  energy0 <- energy |>
    dplyr::select(region, year, prim_ener) |>
    dplyr::distinct(region, year, .keep_all = TRUE)

  gdp0 <- gdp |>
    dplyr::select(region, year, gdp) |>
    dplyr::distinct(region, year, .keep_all = TRUE)

  co20 <- co2 |>
    dplyr::select(region, year, co2) |>
    dplyr::distinct(region, year, .keep_all = TRUE)

  # ---- compute energy intensity at country level ----
  energ_int0 <- energy0 |>
    dplyr::left_join(gdp0, by = c("region","year")) |>
    dplyr::mutate(energ_int = prim_ener / gdp) |>
    dplyr::select(region, year, energ_int)

  # ---- construct ROW series if requested ----
  # Note: sums use na.rm=TRUE, but if *all* values are NA in a year, sum() returns 0.
  # We'll set those to NA afterwards by tracking whether any non-NA existed.
  row_pack <- NULL
  if (isTRUE(include_row)) {
    not_focus <- function(df) df |> dplyr::filter(!region %in% focus_countries)

    sum_with_na_guard <- function(df, value_col) {
      df |>
        dplyr::group_by(year) |>
        dplyr::summarise(
          value = sum(.data[[value_col]], na.rm = TRUE),
          any_non_na = any(!is.na(.data[[value_col]])),
          .groups = "drop"
        ) |>
        dplyr::mutate(value = dplyr::if_else(any_non_na, value, as.numeric(NA))) |>
        dplyr::select(year, value)
    }

    co2_row   <- sum_with_na_guard(not_focus(co20), "co2")       |> dplyr::rename(CO2_ROW = value)
    gdp_row   <- sum_with_na_guard(not_focus(gdp0), "gdp")       |> dplyr::rename(GDP_ROW = value)
    en_row    <- sum_with_na_guard(not_focus(energy0), "prim_ener") |> dplyr::rename(ENERGY_ROW = value)

    energ_int_row <- en_row |>
      dplyr::left_join(gdp_row, by = "year") |>
      dplyr::mutate(ENERG_INT_ROW = ENERGY_ROW / GDP_ROW) |>
      dplyr::select(year, ENERG_INT_ROW)

    row_pack <- list(
      co2_row = co2_row,
      gdp_row = gdp_row,
      energy_row = en_row,
      energ_int_row = energ_int_row
    )
  }

  # ---- build wide "checks" tables for focus countries (handy for sanity checks) ----
  focus_wide <- function(df, value_col, prefix) {
    df |>
      dplyr::filter(region %in% focus_countries) |>
      dplyr::transmute(year, name = paste0(prefix, "_", region), value = .data[[value_col]]) |>
      tidyr::pivot_wider(names_from = name, values_from = value)
  }

  co2_wide <- focus_wide(co20, "co2", "CO2")
  gdp_wide <- focus_wide(gdp0, "gdp", "GDP")
  en_wide  <- focus_wide(energy0, "prim_ener", "ENERGY")
  ei_wide  <- focus_wide(energ_int0, "energ_int", "ENERG_INT")

  # optionally attach ROW columns to the wide checks
  if (isTRUE(include_row)) {
    co2_wide <- co2_wide |> dplyr::left_join(row_pack$co2_row, by = "year")
    gdp_wide <- gdp_wide |> dplyr::left_join(row_pack$gdp_row, by = "year")
    en_wide  <- en_wide  |> dplyr::left_join(row_pack$energy_row, by = "year")
    ei_wide  <- ei_wide  |> dplyr::left_join(row_pack$energ_int_row, by = "year")
  }

  # ---- long local_data for {osem}: time | na_item | values ----
  mk_long <- function(df, value_col, prefix) {
    df |>
      dplyr::transmute(
        time = make_time(year),
        na_item = paste0(prefix, "_", region),
        values = .data[[value_col]]
      )
  }

  local_data <- dplyr::bind_rows(
    mk_long(co20,       "co2",       "CO2"),
    mk_long(gdp0,       "gdp",       "GDP"),
    mk_long(energy0,    "prim_ener", "ENERGY"),
    mk_long(energ_int0, "energ_int", "ENERG_INT")
  )

  if (isTRUE(include_row)) {
    local_data <- dplyr::bind_rows(
      local_data,
      row_pack$co2_row       |> dplyr::transmute(time = make_time(year), na_item = "CO2_ROW",       values = CO2_ROW),
      row_pack$gdp_row       |> dplyr::transmute(time = make_time(year), na_item = "GDP_ROW",       values = GDP_ROW),
      row_pack$energy_row    |> dplyr::transmute(time = make_time(year), na_item = "ENERGY_ROW",    values = ENERGY_ROW),
      row_pack$energ_int_row |> dplyr::transmute(time = make_time(year), na_item = "ENERG_INT_ROW", values = ENERG_INT_ROW)
    )
  }

  # Optional tiny epsilon for zeros if user wants it (off by default)
  if (is.numeric(log_safe_eps) && log_safe_eps > 0) {
    local_data <- local_data |>
      dplyr::mutate(values = dplyr::if_else(values == 0, log_safe_eps, values))
  }

  # ---- build OSEM specification ----
  # CO2_<c> = GDP_<c> + ENERGY_<c> + ENERG_INT_<c>
  spec_rows <- lapply(focus_countries, function(cc) {
    data.frame(
      type = "n",
      dependent = paste0("CO2_", cc),
      independent = paste0("GDP_", cc, " + ENERGY_", cc, " + ENERG_INT_", cc),
      stringsAsFactors = FALSE
    )
  })
  specification <- dplyr::bind_rows(spec_rows)

  if (isTRUE(include_row)) {
    specification <- dplyr::bind_rows(
      specification,
      dplyr::tibble(
        type = "n",
        dependent = "CO2_ROW",
        independent = "GDP_ROW + ENERGY_ROW + ENERG_INT_ROW"
      )
    )
    total_rhs <- paste(c(paste0("CO2_", focus_countries), "CO2_ROW"), collapse = " + ")
  } else {
    total_rhs <- paste(paste0("CO2_", focus_countries), collapse = " + ")
  }

  specification <- dplyr::bind_rows(
    specification,
    dplyr::tibble(type = "d", dependent = "CO2_TOTAL", independent = total_rhs)
  )

  # ---- dictionary (local-only) ----
  all_vars <- unique(c(
    specification$dependent,
    unlist(strsplit(specification$independent, "\\s*\\+\\s*"))
  ))
  all_vars <- trimws(all_vars)
  all_vars <- all_vars[all_vars != ""]
  dictionary <- dplyr::tibble(
    model_varname = all_vars,
    database = dplyr::if_else(all_vars == "CO2_TOTAL", NA, "local"),
    full_name = NA,
    dataset_id = "local",
    freq = "A"
  )

  list(
    local_data = local_data,
    specification = specification,
    dictionary = dictionary,
    wide_checks = list(
      co2 = co2_wide,
      gdp = gdp_wide,
      energy = en_wide,
      energ_int = ei_wide
    )
  )
}
