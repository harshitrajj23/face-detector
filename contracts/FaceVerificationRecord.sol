// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title FaceVerificationRecord
 * @notice Anchors facial biometric hashes and discovered social media post fingerprints
 * to create an immutable, tamper-evident record of identity provenance.
 */
contract FaceVerificationRecord {
    struct Record {
        bytes32 recordId;
        bytes32 faceHash;
        bytes32 postHash;
        string platform;
        string postUrl;
        string author;
        uint256 timestamp;
        address recordedBy;
        bool exists;
    }

    // Mapping from recordId -> Record
    mapping(bytes32 => Record) private _records;
    
    // List of all record IDs
    bytes32[] private _recordIds;

    // Contract Owner / Authority
    address public owner;

    event FaceVerificationRecorded(
        bytes32 indexed recordId,
        bytes32 indexed faceHash,
        bytes32 indexed postHash,
        string platform,
        string postUrl,
        string author,
        address recordedBy,
        uint256 timestamp
    );

    constructor() {
        owner = msg.sender;
    }

    /**
     * @notice Records a new face scan and matching social media post onto the blockchain.
     */
    function recordVerification(
        bytes32 recordId,
        bytes32 faceHash,
        bytes32 postHash,
        string calldata platform,
        string calldata postUrl,
        string calldata author
    ) external returns (bytes32) {
        require(!_records[recordId].exists, "Record already exists with this ID");
        require(faceHash != bytes32(0), "Invalid face hash");
        require(postHash != bytes32(0), "Invalid post hash");

        _records[recordId] = Record({
            recordId: recordId,
            faceHash: faceHash,
            postHash: postHash,
            platform: platform,
            postUrl: postUrl,
            author: author,
            timestamp: block.timestamp,
            recordedBy: msg.sender,
            exists: true
        });

        _recordIds.push(recordId);

        emit FaceVerificationRecorded(
            recordId,
            faceHash,
            postHash,
            platform,
            postUrl,
            author,
            msg.sender,
            block.timestamp
        );

        return recordId;
    }

    /**
     * @notice Re-verifies whether provided hashes match the on-chain recorded fingerprint.
     */
    function verifyRecord(
        bytes32 recordId,
        bytes32 faceHash,
        bytes32 postHash
    ) external view returns (
        bool isValid,
        uint256 timestamp,
        string memory platform,
        string memory postUrl,
        string memory author,
        address recordedBy
    ) {
        Record memory rec = _records[recordId];
        if (!rec.exists) {
            return (false, 0, "", "", "", address(0));
        }

        bool matchFace = (rec.faceHash == faceHash);
        bool matchPost = (rec.postHash == postHash);
        bool valid = matchFace && matchPost;

        return (
            valid,
            rec.timestamp,
            rec.platform,
            rec.postUrl,
            rec.author,
            rec.recordedBy
        );
    }

    /**
     * @notice Retrieves record details by record ID.
     */
    function getRecord(bytes32 recordId) external view returns (Record memory) {
        require(_records[recordId].exists, "Record does not exist");
        return _records[recordId];
    }

    /**
     * @notice Total number of recorded verifications.
     */
    function totalRecords() external view returns (uint256) {
        return _recordIds.length;
    }

    /**
     * @notice Returns record ID at specific index.
     */
    function recordIdAtIndex(uint256 index) external view returns (bytes32) {
        require(index < _recordIds.length, "Index out of bounds");
        return _recordIds[index];
    }
}
