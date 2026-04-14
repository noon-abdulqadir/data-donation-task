"""
Netflix

This module provides an example flow of a Netflix data donation study.

Assumptions:
It handles DDPs in the English language with filetype CSV.
Netflix DDPs may have files nested under a numeric user ID prefix directory.
"""
import logging
import random
from collections import Counter

import pandas as pd
import port.api.d3i_props as d3i_props
import port.api.props as props
import port.helpers.extraction_helpers as eh
import port.helpers.port_helpers as ph
import port.helpers.validate as validate
from port.api.d3i_props import ExtractionResult
from port.helpers.extraction_helpers import ZipArchiveReader
from port.helpers.flow_builder import FlowBuilder
from port.helpers.validate import DDPCategory, DDPFiletype, Language

logger = logging.getLogger(__name__)

DDP_CATEGORIES = [
    DDPCategory(
        id="csv",
        ddp_filetype=DDPFiletype.CSV,
        language=Language.EN,
        known_files=[
            "MyList.csv", "ViewingActivity.csv", "SearchHistory.csv",
            "IndicatedPreferences.csv", "PlaybackRelatedEvents.csv",
            "InteractiveTitles.csv", "Ratings.csv", "GamePlaySession.csv",
            "IpAddressesLogin.csv", "IpAddressesAccountCreation.txt",
            "IpAddressesStreaming.csv", "Additional Information.pdf",
            "MessagesSentByNetflix.csv", "AccountDetails.csv",
            "ProductCancellationSurvey.txt", "CSContact.txt",
            "ChatTranscripts.txt", "Cover Sheet.pdf", "Devices.csv",
            "ParentalControlsRestrictedTitles.txt", "AvatarHistory.csv",
            "Profiles.csv", "Clickstream.csv", "BillingHistory.csv",
            "AccessAndDevices.csv", "ExtraMembers.txt",
        ]
    )
]


def extract_users(reader: ZipArchiveReader) -> list[str]:
    """Extract all profile names from Profiles.csv (first column).

    Falls back to ViewingActivity.csv if Profiles.csv is not available.
    Uses column position rather than name to handle different DDP languages.
    """
    out: list[str] = []

    # Prefer Profiles.csv — dedicated profile list, first column is always profile name
    result = reader.csv("Profiles.csv")
    df = result.data if result.found else pd.DataFrame()

    if df.empty:
        # Fallback: extract unique values from ViewingActivity.csv
        result = reader.csv("ViewingActivity.csv")
        df = result.data if result.found else pd.DataFrame()

    try:
        if not df.empty:
            # Use "Profile Name" if present, otherwise first column
            if "Profile Name" in df.columns:
                out = df["Profile Name"].unique().tolist()
            else:
                out = df[df.columns[0]].unique().tolist()
            out.sort()
    except Exception as e:
        logger.error("Cannot extract users: %s", e)
        reader.errors[type(e).__name__] += 1
    return out


def keep_user(df: pd.DataFrame, selected_user: str) -> pd.DataFrame:
    """Keep only rows where the profile name column matches selected_user.

    Finds the profile column by checking which column contains the selected_user value,
    preferring "Profile Name" if it exists.
    """
    try:
        if "Profile Name" in df.columns:
            df = df.loc[df["Profile Name"] == selected_user].reset_index(drop=True)
        else:
            # Find the column containing the selected user
            for col in df.columns:
                if selected_user in df[col].values:
                    df = df.loc[df[col] == selected_user].reset_index(drop=True)
                    break
    except Exception as e:
        logger.info(e)
    return df


def netflix_to_df(reader: ZipArchiveReader, file_name: str, selected_user: str) -> pd.DataFrame:
    """Load a Netflix CSV, filter to selected user."""
    result = reader.csv(file_name)
    if not result.found:
        return pd.DataFrame()
    return keep_user(result.data, selected_user)


def ratings_to_df(reader: ZipArchiveReader, selected_user: str, errors: Counter) -> pd.DataFrame:
    """Extract ratings — title, thumbs value, timestamp."""
    columns_to_keep = ["Title Name", "Thumbs Value", "Event Utc Ts"]

    df = netflix_to_df(reader, "Ratings.csv", selected_user)

    out = pd.DataFrame()
    try:
        if not df.empty:
            out = pd.DataFrame(df[columns_to_keep])
    except Exception as e:
        logger.error("Data extraction error: %s", e)
        errors[type(e).__name__] += 1

    return out


def time_string_to_hours(time_str: str) -> float:
    try:
        hours, minutes, seconds = map(int, time_str.split(':'))
        total_hours = (hours * 3600 + minutes * 60 + seconds) / 3600
    except Exception:
        return 0.0
    return round(total_hours, 3)


def viewing_activity_to_df(reader: ZipArchiveReader, selected_user: str, errors: Counter) -> pd.DataFrame:
    """Extract viewing activity — start time, duration, title, type."""
    columns_to_keep = ["Start Time", "Duration", "Title", "Supplemental Video Type"]

    df = netflix_to_df(reader, "ViewingActivity.csv", selected_user)
    remove_values = ["TEASER_TRAILER", "HOOK", "TRAILER", "CINEMAGRAPH"]
    out = pd.DataFrame()

    try:
        if not df.empty:
            out = pd.DataFrame(df[columns_to_keep])
            mask = out["Supplemental Video Type"].isin(remove_values)
            out = out[~mask].reset_index(drop=True)
            out["Duration"] = out["Duration"].apply(time_string_to_hours)
            out = out.sort_values(by="Start Time", ascending=True).reset_index(drop=True)
    except Exception as e:
        logger.error("Data extraction error: %s", e)
        errors[type(e).__name__] += 1

    return out


def search_history_to_df(reader: ZipArchiveReader, selected_user: str, errors: Counter) -> pd.DataFrame:
    """Extract search history — query, displayed result, timestamp."""
    df = netflix_to_df(reader, "SearchHistory.csv", selected_user)
    out = pd.DataFrame()

    try:
        if not df.empty:
            columns_to_keep = [c for c in ["Query Typed", "Displayed Name", "Utc Timestamp"] if c in df.columns]
            out = pd.DataFrame(df[columns_to_keep])
            if "Utc Timestamp" in out.columns:
                out = out.sort_values(by="Utc Timestamp", ascending=False).reset_index(drop=True)
    except Exception as e:
        logger.error("Data extraction error: %s", e)
        errors[type(e).__name__] += 1

    return out


def extraction(reader: ZipArchiveReader, selected_user: str) -> ExtractionResult:
    errors = reader.errors
    tables = [
        d3i_props.PropsUIPromptConsentFormTableViz(
            id="netflix_ratings",
            data_frame=ratings_to_df(reader, selected_user, errors),
            title=props.Translatable({
                "en": "Your ratings on Netflix",
                "nl": "Uw beoordelingen op Netflix",
            }),
            description=props.Translatable({
                "en": "Titles you have rated on Netflix.",
                "nl": "Titels die u op Netflix heeft beoordeeld.",
            }),
            headers={
                "Title Name": props.Translatable({"en": "Title", "nl": "Titel"}),
                "Thumbs Value": props.Translatable({"en": "Thumbs value", "nl": "Aantal duimpjes omhoog"}),
                "Event Utc Ts": props.Translatable({"en": "Date", "nl": "Datum en tijd"}),
            },
            visualizations=[
                {
                    "title": {
                        "en": "Titles rated by thumbs value",
                        "nl": "Beoordeelde titels op basis van duimpjes",
                    },
                    "type": "wordcloud",
                    "textColumn": "Title Name",
                    "valueColumn": "Thumbs Value",
                },
            ],
        ),
        d3i_props.PropsUIPromptConsentFormTableViz(
            id="netflix_viewing_activity",
            data_frame=viewing_activity_to_df(reader, selected_user, errors),
            title=props.Translatable({
                "en": "What you watched",
                "nl": "Wat u heeft gekeken",
            }),
            description=props.Translatable({
                "en": "This table shows what titles you watched, when, and for how long.",
                "nl": "Deze tabel toont welke titels u heeft gekeken, wanneer, en hoe lang.",
            }),
            headers={
                "Start Time": props.Translatable({"en": "Start time", "nl": "Starttijd"}),
                "Duration": props.Translatable({"en": "Hours watched", "nl": "Aantal uur gekeken"}),
                "Title": props.Translatable({"en": "Title", "nl": "Titel"}),
                "Supplemental Video Type": props.Translatable({"en": "Type", "nl": "Aanvullende informatie"}),
            },
            visualizations=[
                {
                    "title": {
                        "en": "Total hours watched per month",
                        "nl": "Totaal aantal uren gekeken per maand",
                    },
                    "type": "area",
                    "group": {
                        "column": "Start Time",
                        "dateFormat": "month",
                        "label": "Month",
                    },
                    "values": [{
                        "column": "Duration",
                        "aggregate": "sum",
                    }],
                },
                {
                    "title": {
                        "en": "Total hours watched by hour of the day",
                        "nl": "Totaal aantal uur gekeken per uur van de dag",
                    },
                    "type": "bar",
                    "group": {
                        "column": "Start Time",
                        "dateFormat": "hour_cycle",
                    },
                    "values": [{
                        "column": "Duration",
                        "aggregate": "sum",
                    }],
                },
            ],
        ),
        d3i_props.PropsUIPromptConsentFormTableViz(
            id="netflix_search_history",
            data_frame=search_history_to_df(reader, selected_user, errors),
            title=props.Translatable({
                "en": "Your search history on Netflix",
                "nl": "Uw zoekgeschiedenis op Netflix",
            }),
            description=props.Translatable({
                "en": "Searches you have performed on Netflix.",
                "nl": "Zoekopdrachten die u op Netflix heeft uitgevoerd.",
            }),
            headers={
                "Query Typed": props.Translatable({"en": "Search query", "nl": "Zoekterm"}),
                "Displayed Name": props.Translatable({"en": "Result shown", "nl": "Weergegeven resultaat"}),
                "Utc Timestamp": props.Translatable({"en": "Date", "nl": "Datum en tijd"}),
            },
            visualizations=[
                {
                    "title": {
                        "en": "Most searched terms",
                        "nl": "Meest gezochte termen",
                    },
                    "type": "wordcloud",
                    "textColumn": "Query Typed",
                    "tokenize": False,
                },
            ],
        ),
    ]

    return ExtractionResult(
        tables=[table for table in tables if not table.data_frame.empty],
        errors=errors,
    )


class NetflixFlow(FlowBuilder):
    def __init__(self, session_id: str):
        super().__init__(session_id, "Netflix")
        self.titles_for_questionnaire: list[str] = []

    def validate_file(self, file):
        return validate.validate_zip(DDP_CATEGORIES, file)

    def extract_titles_list(self, results: ExtractionResult):
        try:
            for table in results.tables:
                if table.id == "netflix_viewing_activity" and not table.data_frame.empty:
                    self.titles_for_questionnaire = table.data_frame["Title"].dropna().unique().tolist()

        except Exception as e:
            logger.error("Error extracting title for questionnaire: %s", e)

    def extract_data(self, file, validation):
        errors = Counter()
        reader = ZipArchiveReader(file, validation.archive_members, errors)
        selected_user = ""
        users = extract_users(reader)

        if len(users) == 1:
            selected_user = users[0]

            # return extraction(reader, selected_user)
        elif len(users) > 1:
            title = props.Translatable({
                "en": "Select your Netflix profile name",
                "nl": "Kies jouw Netflix profielnaam",
            })
            empty_text = props.Translatable({"en": "", "nl": ""})
            radio_prompt = ph.generate_radio_prompt(title, empty_text, users)
            selection = yield ph.render_page(empty_text, radio_prompt)
            selected_user = selection.value
            # return extraction(reader, selected_user)

        # Save a list of titles here for questionnaire later
        results = extraction(reader, selected_user)

        self.extract_titles_list(results)

        return results

    # Netflix-only open-ended question about randomly picked title watched by user
    # Function based on generate_questionnaire in port_helpers.py
    def get_title_for_questionnaire(self) -> str | None:
        try:
            if self.titles_for_questionnaire:
                return random.choice(self.titles_for_questionnaire)
        except Exception as e:
            logger.error("get_title_for_questionnaire error: %s", e)


    def open_ended_questionnaire(self):
        try:
            title = self.get_title_for_questionnaire()
            if not title:
                return
            questionnaire = ph.generate_questionnaire()
            questionnaire.description = props.Translatable({
                'en': "We'd like to ask you about a title from your viewing history.",
                'nl': "We willen u vragen naar een titel uit uw kijkgeschiedenis.",
            })
            questionnaire.questions = [
                d3i_props.PropsUIQuestionOpen(
                    id="netflix_recall",
                    question=props.Translatable({
                        'en': f'What do you remember about watching {title}?',
                        'nl': f'Wat herinner je je van het kijken naar {title}?',
                    })
                )
            ]
            result = yield ph.render_page(
                props.Translatable({'en': 'Open ended question', 'nl': 'Open vraag'}),
                questionnaire,
            )

            answers = pd.read_json(result.value, typ='series')
            df = pd.DataFrame([{
                'session_id': self.session_id,
                'Title': title,
                'Open-ended Answer': answers.get('netflix_recall', ''),
            }])
            yield ph.donate(f'{self.session_id}-netflix-recall', df.to_json())

        except Exception as e:
            logger.error('open_ended_questionnaire error: %s', e)


    def post_donate_flow(self):
        """Run the recall questionnaire after successful donation."""
        return self.open_ended_questionnaire()


def process(session_id):
    flow = NetflixFlow(session_id)
    return flow.start_flow()
