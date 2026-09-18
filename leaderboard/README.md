# URB Leaderboard

This folder contains a static leaderboard generator for URB experiment results.

The generated page has three native views. The Home view is kept in
`home_content.html`, `home_styles.css`, and `home_script.js`; the generator embeds
them into the same self-contained page as the leaderboard and studies views.

## Generate the page

From repository root:

```bash
python3 leaderboard/generate_leaderboard.py
```

Default output:

- `docs/leaderboard/index.html`
- `docs/leaderboard/urb.png`
- `docs/leaderboard/study-covers/*.png` for published studies with a cover

## Options

- `--results-dir`: source directory with run folders (default: `results`)
- `--output-dir`: output site directory (default: `docs/leaderboard`)
- `--studies-dir`: source directory containing studies (default: `studies`)
- `--repo-url`: optional GitHub URL prefix for experiment links, e.g. `https://github.com/COeXISTENCE-PROJECT/URB/tree/main/`; when omitted, it is inferred from `title_link_url` in `leaderboard_strings.json`
- `--local-link-prefix`: relative fallback prefix for links when `--repo-url` is not provided (default: `..`)

## Included behavior

- Indexes completed run folders with finite `t_test` and `t_CAV` metrics
- Shows studies with a proposal, experiment record, and final report; missing reviews are marked
- Sortable metric columns
- Network tabs, plus filters for environment config, task config, environment seed, and optional project
- Collapsing reruns with identical non-seed settings into averaged rows after filtering, so
  selecting a seed produces the correctly averaged subgroup
- Hover details listing the concrete seeds behind a collapsed `varies` value
- Sorting by experiment date, using the result's first Git commit when available
- Metric descriptions as column tooltips
- CSV export of the visible table
