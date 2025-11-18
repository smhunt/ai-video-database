"""Video embedding service for semantic search using Qdrant."""
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import numpy as np
import os
from openai import OpenAI


class VideoEmbeddingService:
    """Manages video frame embeddings in Qdrant for semantic search."""

    COLLECTION_NAME = "video_frames"
    VECTOR_SIZE = 1024  # mxbai-embed-large-v1 dimension
    DISTANCE_METRIC = Distance.COSINE

    def __init__(self, qdrant_url: str = None):
        """Initialize Qdrant client and ensure collection exists."""
        if qdrant_url is None:
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(url=qdrant_url)
        self._ensure_collection()
        logger.info(f"VideoEmbeddingService initialized with collection {self.COLLECTION_NAME}")

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.COLLECTION_NAME not in collection_names:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE, distance=self.DISTANCE_METRIC
                ),
            )
            logger.info(f"Created Qdrant collection: {self.COLLECTION_NAME}")
        else:
            logger.info(f"Qdrant collection already exists: {self.COLLECTION_NAME}")

    def embed_frame_descriptions(
        self, descriptions: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for frame descriptions using OpenAI API.

        Args:
            descriptions: List of frame descriptions

        Returns:
            List of embedding vectors
        """
        if not descriptions:
            return []

        logger.info(f"Generating embeddings for {len(descriptions)} descriptions")

        # Use OpenAI embeddings API
        client = OpenAI()
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=descriptions,
            dimensions=1024
        )

        embeddings = [item.embedding for item in response.data]
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

    def index_frames(
        self,
        video_id: int,
        frame_data: List[Dict[str, Any]],
    ):
        """
        Index frame descriptions and metadata in Qdrant.

        Args:
            video_id: Database ID of the video
            frame_data: List of dicts with:
                - frame_id: Database frame ID
                - description: Frame description text
                - timestamp_seconds: Time in video
                - objects: List of objects
                - actions: List of actions
                - scene_type: Scene category
                - excitement_score: 1-10 rating
        """
        if not frame_data:
            logger.warning("No frames to index")
            return

        logger.info(f"Indexing {len(frame_data)} frames for video {video_id}")

        # Generate embeddings for descriptions
        descriptions = [frame["description"] for frame in frame_data]
        embeddings = self.embed_frame_descriptions(descriptions)

        # Create points for Qdrant
        points = []
        for i, (frame, embedding) in enumerate(zip(frame_data, embeddings)):
            point = PointStruct(
                id=frame["frame_id"],  # Use database frame ID as point ID
                vector=embedding,
                payload={
                    "video_id": video_id,
                    "frame_id": frame["frame_id"],
                    "timestamp_seconds": frame["timestamp_seconds"],
                    "description": frame["description"],
                    "objects": frame.get("objects", []),
                    "actions": frame.get("actions", []),
                    "scene_type": frame.get("scene_type", "unknown"),
                    "excitement_score": frame.get("excitement_score", 5),
                },
            )
            points.append(point)

        # Upsert to Qdrant
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points,
        )

        logger.info(f"Successfully indexed {len(points)} frames")

    def search_frames(
        self,
        query: str,
        video_id: Optional[int] = None,
        limit: int = 10,
        excitement_threshold: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for frames matching query.

        Args:
            query: Search query text
            video_id: Optional filter by video ID
            limit: Maximum number of results
            excitement_threshold: Optional minimum excitement score

        Returns:
            List of matching frames with metadata and scores
        """
        # Generate query embedding
        query_embedding = self.embed_frame_descriptions([query])[0]

        # Build filter conditions
        filter_conditions = []
        if video_id is not None:
            filter_conditions.append(
                FieldCondition(key="video_id", match=MatchValue(value=video_id))
            )
        if excitement_threshold is not None:
            filter_conditions.append(
                FieldCondition(
                    key="excitement_score",
                    range={"gte": excitement_threshold},
                )
            )

        # Build filter object
        search_filter = None
        if filter_conditions:
            search_filter = Filter(must=filter_conditions)

        # Search Qdrant
        results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "frame_id": result.payload["frame_id"],
                    "video_id": result.payload["video_id"],
                    "timestamp_seconds": result.payload["timestamp_seconds"],
                    "description": result.payload["description"],
                    "objects": result.payload["objects"],
                    "actions": result.payload["actions"],
                    "scene_type": result.payload["scene_type"],
                    "excitement_score": result.payload["excitement_score"],
                    "similarity_score": result.score,
                }
            )

        logger.info(f"Found {len(formatted_results)} frames matching '{query}'")
        return formatted_results

    def get_exciting_frames(
        self,
        video_id: int,
        threshold: int = 7,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get frames with high excitement scores.

        Args:
            video_id: Video database ID
            threshold: Minimum excitement score
            limit: Maximum number of results

        Returns:
            List of exciting frames sorted by score
        """
        # Search with filter but use a neutral query
        # We'll just filter by excitement and return top results
        results = self.client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="video_id", match=MatchValue(value=video_id)),
                    FieldCondition(
                        key="excitement_score",
                        range={"gte": threshold},
                    ),
                ]
            ),
            limit=limit,
            with_payload=True,
        )

        # Format and sort by excitement score
        formatted_results = []
        for point in results[0]:  # scroll returns (points, next_page_offset)
            formatted_results.append(
                {
                    "frame_id": point.payload["frame_id"],
                    "video_id": point.payload["video_id"],
                    "timestamp_seconds": point.payload["timestamp_seconds"],
                    "description": point.payload["description"],
                    "objects": point.payload["objects"],
                    "actions": point.payload["actions"],
                    "scene_type": point.payload["scene_type"],
                    "excitement_score": point.payload["excitement_score"],
                }
            )

        # Sort by excitement score descending
        formatted_results.sort(key=lambda x: x["excitement_score"], reverse=True)

        logger.info(
            f"Found {len(formatted_results)} exciting frames for video {video_id}"
        )
        return formatted_results

    def get_frames_for_video(self, video_id: int) -> List[Dict[str, Any]]:
        """
        Get all indexed frames for a video.

        Args:
            video_id: Video database ID

        Returns:
            List of all frames for the video
        """
        results = self.client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
            ),
            limit=1000,  # Adjust if videos have more frames
            with_payload=True,
        )

        formatted_results = []
        for point in results[0]:
            formatted_results.append(
                {
                    "frame_id": point.payload["frame_id"],
                    "video_id": point.payload["video_id"],
                    "timestamp_seconds": point.payload["timestamp_seconds"],
                    "description": point.payload["description"],
                    "objects": point.payload["objects"],
                    "actions": point.payload["actions"],
                    "scene_type": point.payload["scene_type"],
                    "excitement_score": point.payload["excitement_score"],
                }
            )

        # Sort by timestamp
        formatted_results.sort(key=lambda x: x["timestamp_seconds"])

        logger.info(f"Retrieved {len(formatted_results)} frames for video {video_id}")
        return formatted_results

    def delete_video_frames(self, video_id: int):
        """
        Delete all frames for a video from the index.

        Args:
            video_id: Video database ID
        """
        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
            ),
        )
        logger.info(f"Deleted all frames for video {video_id}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        info = self.client.get_collection(collection_name=self.COLLECTION_NAME)
        return {
            "total_frames": info.points_count,
            "vector_size": self.VECTOR_SIZE,
            "distance_metric": self.DISTANCE_METRIC.value,
        }
