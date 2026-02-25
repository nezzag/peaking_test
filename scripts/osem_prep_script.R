#' Prepare local multi-country (+WORLD-RESIDUAL) dataset + OSEM spec + dictionary
#'
#' This version constructs the residual block as:
#'   X_RESID = X_WORLD - sum(X_focus)
#' for X in {CO2, GDP, ENERGY}. Energy intensity for residual is computed as ENERGY_RESID/GDP_RESID.
#'
#' Assumption: the world aggregate is encoded as region == "World" in each input dataset.
#'
#' @param energy A tibble with columns: region, region_name, year, prim_ener (and others ok)
#' @param gdp    A tibble with columns: region, region_name, year, gdp (and others ok)
#' @param co2    A tibble with columns: region, region_name, year, co2 (and others ok)
#' @param focus_countries Character vector of ISO3 country codes to model separately
#' @param world_code Character; region code used for world aggregate (default "World")
#' @param resid_name Character; suffix/name for residual block (default "RESID")
#' @param include_resid Logical; if TRUE create the residual block
#' @param year_to_date Logical; if TRUE convert numeric year to Date as YYYY-01-01
#'                     (recommended for {osem}; it expects a Date 'time' column)
#' @param log_safe_eps Small positive number to replace exact zeros in values (off by default)
#'
#' @return A list with:
#'   - local_data: long tibble with columns time, na_item, values
#'   - specification: osem spec table (type/dependent/independent)
#'   - dictionary: minimal dictionary marking all vars as local/identity (your schema)
#'   - wide_checks: useful wide tibbles for debugging
prepare_osem_co2_block <- function(
    energy,
    gdp,
    co2,
    focus_countries = c("DEU","USA","CHN"),
    world_code = "WLD",
    resid_name = "RESID",
    include_resid = TRUE,
    year_to_date = TRUE,
    log_safe_eps = 0
) {
  stopifnot(is.data.frame(energy), is.data.frame(gdp), is.data.frame(co2))
  stopifnot(is.character(focus_countries), length(focus_countries) >= 1)
  stopifnot(is.character(world_code), length(world_code) == 1)
  stopifnot(is.character(resid_name), length(resid_name) == 1)

  # ---- helpers ----
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
    as.Date(sprintf("%04d-01-01", as.integer(y)))
  }

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

  # ---- validate columns ----
  req_cols(energy, c("region","year","prim_ener"), "energy")
  req_cols(gdp,    c("region","year","gdp"),      "gdp")
  req_cols(co2,    c("region","year","co2"),      "co2")

  # ---- minimal, unique region-year ----
  energy0 <- energy |>
    dplyr::select(region, year, prim_ener) |>
    dplyr::distinct(region, year, .keep_all = TRUE)

  gdp0 <- gdp |>
    dplyr::select(region, year, gdp) |>
    dplyr::distinct(region, year, .keep_all = TRUE)

  co20 <- co2 |>
    dplyr::select(region, year, co2) |>
    dplyr::distinct(region, year, .keep_all = TRUE)

  # ---- energy intensity per region (including world + focus countries) ----
  energ_int0 <- energy0 |>
    dplyr::left_join(gdp0, by = c("region","year")) |>
    dplyr::mutate(energ_int = prim_ener / gdp) |>
    dplyr::select(region, year, energ_int)

  # ---- build WORLD minus focus residual if requested ----
  resid_pack <- NULL
  if (isTRUE(include_resid)) {
    # extract world series
    world_co2 <- co20  |> dplyr::filter(region == world_code) |> dplyr::select(year, co2)       |> dplyr::rename(CO2_WORLD = co2)
    world_gdp <- gdp0  |> dplyr::filter(region == world_code) |> dplyr::select(year, gdp)       |> dplyr::rename(GDP_WORLD = gdp)
    world_en  <- energy0 |> dplyr::filter(region == world_code) |> dplyr::select(year, prim_ener) |> dplyr::rename(ENERGY_WORLD = prim_ener)

    # sum focus countries
    focus_co2_sum <- sum_with_na_guard(co20  |> dplyr::filter(region %in% focus_countries), "co2")       |> dplyr::rename(CO2_FOCUS = value)
    focus_gdp_sum <- sum_with_na_guard(gdp0  |> dplyr::filter(region %in% focus_countries), "gdp")       |> dplyr::rename(GDP_FOCUS = value)
    focus_en_sum  <- sum_with_na_guard(energy0 |> dplyr::filter(region %in% focus_countries), "prim_ener") |> dplyr::rename(ENERGY_FOCUS = value)

    # align years and compute residuals
    resid_df <- world_co2 |>
      dplyr::full_join(world_gdp, by = "year") |>
      dplyr::full_join(world_en,  by = "year") |>
      dplyr::full_join(focus_co2_sum, by = "year") |>
      dplyr::full_join(focus_gdp_sum, by = "year") |>
      dplyr::full_join(focus_en_sum,  by = "year") |>
      dplyr::mutate(
        CO2_RESID    = CO2_WORLD    - CO2_FOCUS,
        GDP_RESID    = GDP_WORLD    - GDP_FOCUS,
        ENERGY_RESID = ENERGY_WORLD - ENERGY_FOCUS,
        ENERG_INT_RESID = ENERGY_RESID / GDP_RESID
      ) |>
      dplyr::select(year, CO2_RESID, GDP_RESID, ENERGY_RESID, ENERG_INT_RESID,
                    CO2_WORLD, GDP_WORLD, ENERGY_WORLD, CO2_FOCUS, GDP_FOCUS, ENERGY_FOCUS)

    # rename residual variables to chosen resid_name (default RESID)
    resid_pack <- resid_df |>
      dplyr::rename(
        !!paste0("CO2_", resid_name) := CO2_RESID,
        !!paste0("GDP_", resid_name) := GDP_RESID,
        !!paste0("ENERGY_", resid_name) := ENERGY_RESID,
        !!paste0("ENERG_INT_", resid_name) := ENERG_INT_RESID
      )
  }

  # ---- wide checks for focus countries ----
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

  # also attach world + residual checks
  world_checks <- list(
    co2_world = co20  |> dplyr::filter(region == world_code) |> dplyr::select(year, co2) |> dplyr::rename(CO2_WORLD = co2),
    gdp_world = gdp0  |> dplyr::filter(region == world_code) |> dplyr::select(year, gdp) |> dplyr::rename(GDP_WORLD = gdp),
    en_world  = energy0 |> dplyr::filter(region == world_code) |> dplyr::select(year, prim_ener) |> dplyr::rename(ENERGY_WORLD = prim_ener)
  )

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

  # add residual block as additional "local" series
  if (isTRUE(include_resid)) {
    local_data <- dplyr::bind_rows(
      local_data,
      resid_pack |>
        dplyr::transmute(time = make_time(year), na_item = paste0("CO2_", resid_name), values = .data[[paste0("CO2_", resid_name)]]),
      resid_pack |>
        dplyr::transmute(time = make_time(year), na_item = paste0("GDP_", resid_name), values = .data[[paste0("GDP_", resid_name)]]),
      resid_pack |>
        dplyr::transmute(time = make_time(year), na_item = paste0("ENERGY_", resid_name), values = .data[[paste0("ENERGY_", resid_name)]]),
      resid_pack |>
        dplyr::transmute(time = make_time(year), na_item = paste0("ENERG_INT_", resid_name), values = .data[[paste0("ENERG_INT_", resid_name)]])
    )
  }

  # Optional epsilon for zeros
  if (is.numeric(log_safe_eps) && log_safe_eps > 0) {
    local_data <- local_data |>
      dplyr::mutate(values = dplyr::if_else(values == 0, log_safe_eps, values))
  }

  # ---- OSEM specification ----
  spec_rows <- lapply(focus_countries, function(cc) {
    data.frame(
      type = "n",
      dependent = paste0("CO2_", cc),
      independent = paste0("GDP_", cc, " + ENERGY_", cc, " + ENERG_INT_", cc),
      stringsAsFactors = FALSE
    )
  })
  specification <- dplyr::bind_rows(spec_rows)

  if (isTRUE(include_resid)) {
    specification <- dplyr::bind_rows(
      specification,
      dplyr::tibble(
        type = "n",
        dependent = paste0("CO2_", resid_name),
        independent = paste0("GDP_", resid_name, " + ENERGY_", resid_name, " + ENERG_INT_", resid_name)
      )
    )

    total_rhs <- paste(c(paste0("CO2_", focus_countries), paste0("CO2_", resid_name)), collapse = " + ")
  } else {
    total_rhs <- paste(paste0("CO2_", focus_countries), collapse = " + ")
  }

  specification <- dplyr::bind_rows(
    specification,
    dplyr::tibble(type = "d", dependent = "CO2_TOTAL", independent = total_rhs)
  )

  # ---- dictionary (local-only, using your schema) ----
  rhs_vars <- unlist(strsplit(paste(specification$independent, collapse = " + "), "\\s*\\+\\s*"))
  all_vars <- unique(c(specification$dependent, trimws(rhs_vars)))
  all_vars <- all_vars[all_vars != ""]
  dictionary <- dplyr::tibble(
    model_varname = all_vars,
    database = dplyr::if_else(all_vars == "CO2_TOTAL", NA, "local"),
    full_name = NA_character_,
    dataset_id = "local",
    freq = "A"
  )

  list(
    local_data = local_data,
    specification = specification,
    dictionary = dictionary,
    wide_checks = list(
      focus = list(
        co2 = co2_wide,
        gdp = gdp_wide,
        energy = en_wide,
        energ_int = ei_wide
      ),
      world = world_checks,
      resid = resid_pack
    )
  )
}
