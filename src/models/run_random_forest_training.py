#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import json

from src.models.train_random_forest import (
    train_random_forest,
)


# In[ ]:


#pass Vertex pipeline parameters to containers
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-id",
        required=True,
    )

    parser.add_argument(
        "--feature-table",
        required=True,
    )

    parser.add_argument(
        "--feedback-table",
        required=True,
    )

    parser.add_argument(
        "--feature-columns",
        required=True,
    )

    parser.add_argument(
        "--model-output-uri",
        required=True,
    )

    args = parser.parse_args()

    feature_columns = json.loads(
        args.feature_columns
    )

    result = train_random_forest(
        project_id=args.project_id,
        feature_table=args.feature_table,
        feedback_table=args.feedback_table,
        feature_columns=feature_columns,
        model_output_uri=args.model_output_uri,
    )

    print("\nTraining result:")
    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

