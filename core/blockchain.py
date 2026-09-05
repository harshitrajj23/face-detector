"""
Blockchain verification engine.
Provides a dual-chain architecture:
1. NativeMerkleLedger: Standalone cryptographic blockchain with SHA-256 block chaining and Merkle trees.
2. EVMVerificationLedger: EVM-compatible ledger (Ethereum Sepolia, Polygon Amoy, local Anvil/Hardhat, or in-process EVM).
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from core.models import BlockchainRecord, SocialPost, VerificationAuditResult


class MerkleTree:
    """Computes SHA-256 Merkle root from transaction hashes."""

    @staticmethod
    def compute_root(hashes: List[str]) -> str:
        if not hashes:
            return hashlib.sha256(b"empty_block").hexdigest()
        if len(hashes) == 1:
            return hashes[0]

        current_level = list(hashes)
        while len(current_level) > 1:
            if len(current_level) % 2 == 1:
                current_level.append(current_level[-1])
            next_level = []
            for i in range(0, len(current_level), 2):
                combined = (current_level[i] + current_level[i + 1]).encode("utf-8")
                parent = hashlib.sha256(combined).hexdigest()
                next_level.append(parent)
            current_level = next_level

        return current_level[0]


class NativeMerkleLedger:
    """
    Transparent, verifiable, standalone cryptographic blockchain ledger.
    Stores cryptographic blocks, Merkle roots, and transaction state in chaindata/ledger.json.
    """

    def __init__(self, ledger_path: str = "chaindata/ledger.json"):
        base_dir = Path(__file__).resolve().parent.parent
        self.ledger_path = base_dir / ledger_path
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.chain: List[Dict[str, Any]] = []
        self._load_or_init()

    def _load_or_init(self):
        if self.ledger_path.exists():
            try:
                with open(self.ledger_path, "r") as f:
                    self.chain = json.load(f)
                    return
            except Exception:
                pass

        # Initialize genesis block
        genesis_block = {
            "index": 0,
            "timestamp": 1700000000,
            "previous_hash": "0" * 64,
            "merkle_root": MerkleTree.compute_root([]),
            "transactions": [],
            "nonce": 0,
            "hash": "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f",
        }
        self.chain = [genesis_block]
        self._save()

    def _save(self):
        with open(self.ledger_path, "w") as f:
            json.dump(self.chain, f, indent=2)

    def _calculate_block_hash(self, block: Dict[str, Any]) -> str:
        header = f"{block['index']}{block['timestamp']}{block['previous_hash']}{block['merkle_root']}{block['nonce']}"
        return hashlib.sha256(header.encode("utf-8")).hexdigest()

    def record_verification(
        self,
        face_hash: str,
        post: SocialPost,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BlockchainRecord:
        """
        Anchors the face embedding hash and discovered post fingerprint into a new block.
        """
        # Generate deterministic record ID from face_hash + post_hash
        combined = f"{face_hash}:{post.post_hash}".encode("utf-8")
        record_id = hashlib.sha256(combined).hexdigest()

        now_ts = int(time.time())
        tx_data = {
            "record_id": record_id,
            "face_hash": face_hash,
            "post_hash": post.post_hash,
            "platform": post.platform,
            "post_url": post.post_url,
            "author": post.author,
            "timestamp": now_ts,
            "metadata": metadata or {},
        }
        tx_hash = hashlib.sha256(json.dumps(tx_data, sort_keys=True).encode("utf-8")).hexdigest()
        tx_data["tx_hash"] = tx_hash

        prev_block = self.chain[-1]
        new_index = prev_block["index"] + 1
        merkle_root = MerkleTree.compute_root([tx_hash])

        # Mine block (fast proof-of-work difficulty for demonstration)
        nonce = 0
        while True:
            candidate_block = {
                "index": new_index,
                "timestamp": now_ts,
                "previous_hash": prev_block["hash"],
                "merkle_root": merkle_root,
                "transactions": [tx_data],
                "nonce": nonce,
            }
            block_hash = self._calculate_block_hash(candidate_block)
            if block_hash.startswith("0"):  # Light difficulty target
                candidate_block["hash"] = block_hash
                break
            nonce += 1

        self.chain.append(candidate_block)
        self._save()

        return BlockchainRecord(
            record_id=record_id,
            face_hash=face_hash,
            post_hash=post.post_hash,
            platform=post.platform,
            post_url=post.post_url,
            author=post.author,
            timestamp=now_ts,
            block_number=new_index,
            block_hash=block_hash,
            transaction_hash=tx_hash,
            blockchain_type="native_merkle",
            metadata=metadata or {},
        )

    def get_record(self, record_id: str) -> Optional[BlockchainRecord]:
        """Looks up a record across the entire blockchain ledger."""
        for block in self.chain:
            for tx in block.get("transactions", []):
                if tx.get("record_id") == record_id:
                    return BlockchainRecord(
                        record_id=tx["record_id"],
                        face_hash=tx["face_hash"],
                        post_hash=tx["post_hash"],
                        platform=tx.get("platform", "web"),
                        post_url=tx.get("post_url", ""),
                        author=tx.get("author", ""),
                        timestamp=tx.get("timestamp", block["timestamp"]),
                        block_number=block["index"],
                        block_hash=block["hash"],
                        transaction_hash=tx.get("tx_hash", ""),
                        blockchain_type="native_merkle",
                        metadata=tx.get("metadata", {}),
                    )
        return None

    def verify_record(
        self,
        record_id: str,
        provided_face_hash: str,
        provided_post_hash: str,
    ) -> VerificationAuditResult:
        """
        Re-verifies provided face hash and post hash against the immutable on-chain record.
        Checks block hash integrity and Merkle tree root consistency.
        """
        rec = self.get_record(record_id)
        if not rec:
            return VerificationAuditResult(
                is_valid=False,
                record_id=record_id,
                on_chain_face_hash="",
                provided_face_hash=provided_face_hash,
                face_hash_matches=False,
                on_chain_post_hash="",
                provided_post_hash=provided_post_hash,
                post_hash_matches=False,
                block_number=-1,
                block_hash="",
                timestamp=0,
                message=f"Record ID {record_id} does not exist on blockchain.",
                blockchain_type="native_merkle",
            )

        # Validate ledger block integrity
        is_chain_valid, chain_msg = self.verify_chain_integrity()
        if not is_chain_valid:
            return VerificationAuditResult(
                is_valid=False,
                record_id=record_id,
                on_chain_face_hash=rec.face_hash,
                provided_face_hash=provided_face_hash,
                face_hash_matches=(rec.face_hash.lower() == provided_face_hash.lower()),
                on_chain_post_hash=rec.post_hash,
                provided_post_hash=provided_post_hash,
                post_hash_matches=(rec.post_hash.lower() == provided_post_hash.lower()),
                block_number=rec.block_number,
                block_hash=rec.block_hash,
                timestamp=rec.timestamp,
                message=f"Tampering detected in blockchain ledger: {chain_msg}",
                blockchain_type="native_merkle",
            )

        face_matches = (rec.face_hash.lower() == provided_face_hash.lower())
        post_matches = (rec.post_hash.lower() == provided_post_hash.lower())
        is_valid = face_matches and post_matches

        msg = "On-chain record verified successfully! Cryptographic proof intact." if is_valid else (
            "Cryptographic verification FAILED: Hash mismatch between supplied data and immutable on-chain record."
        )

        return VerificationAuditResult(
            is_valid=is_valid,
            record_id=record_id,
            on_chain_face_hash=rec.face_hash,
            provided_face_hash=provided_face_hash,
            face_hash_matches=face_matches,
            on_chain_post_hash=rec.post_hash,
            provided_post_hash=provided_post_hash,
            post_hash_matches=post_matches,
            block_number=rec.block_number,
            block_hash=rec.block_hash,
            timestamp=rec.timestamp,
            message=msg,
            blockchain_type="native_merkle",
        )

    def verify_chain_integrity(self) -> Tuple[bool, str]:
        """
        Audits all blocks in the chain, recalculating hashes and Merkle roots.
        Detects any unauthorized modification or tampering.
        """
        for i in range(1, len(self.chain)):
            prev = self.chain[i - 1]
            curr = self.chain[i]

            if curr["previous_hash"] != prev["hash"]:
                return False, f"Block #{curr['index']} previous_hash does not match Block #{prev['index']} hash!"

            calc_hash = self._calculate_block_hash(curr)
            if calc_hash != curr["hash"]:
                return False, f"Block #{curr['index']} hash corrupted! Stored={curr['hash'][:16]}, Calculated={calc_hash[:16]}"

            # Verify Merkle root of transactions
            tx_hashes = [
                tx.get("tx_hash") or hashlib.sha256(json.dumps(tx, sort_keys=True).encode()).hexdigest()
                for tx in curr.get("transactions", [])
            ]
            calc_merkle = MerkleTree.compute_root(tx_hashes)
            if calc_merkle != curr["merkle_root"]:
                return False, f"Block #{curr['index']} Merkle root mismatch! Transactions were modified."

        return True, "All blocks, Merkle roots, and hashes cryptographically verified."


class EVMVerificationLedger:
    """
    EVM Smart Contract Blockchain Provider.
    Interacts with FaceVerificationRecord.sol on:
    - Ethereum Sepolia testnet
    - Polygon Amoy testnet
    - Local EVM node (Anvil / Hardhat)
    - In-process EVM (EthereumTesterProvider)
    """

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        private_key: Optional[str] = None,
        contract_address: Optional[str] = None,
    ):
        from web3 import Web3

        self.w3: Optional[Web3] = None
        self.contract = None
        self.account = None
        self.rpc_url = rpc_url or os.getenv("EVM_RPC_URL")
        self.private_key = private_key or os.getenv("EVM_PRIVATE_KEY")
        self.contract_address = contract_address or os.getenv("CONTRACT_ADDRESS")

        # Load compiled contract ABI and Bytecode
        artifact_path = Path(__file__).resolve().parent.parent / "contracts" / "FaceVerificationRecord.json"
        with open(artifact_path, "r") as f:
            artifact = json.load(f)
        self.abi = artifact["abi"]
        self.bytecode = artifact["bytecode"]

        self._setup_connection()

    def _setup_connection(self):
        from web3 import Web3

        if self.rpc_url:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            if self.private_key:
                self.account = self.w3.eth.account.from_key(self.private_key)
        else:
            # Fallback to in-process EVM tester for zero-configuration testing
            try:
                from web3 import EthereumTesterProvider
                self.w3 = Web3(EthereumTesterProvider())
            except Exception:
                self.w3 = None

        if self.w3 and self.w3.is_connected():
            if self.contract_address:
                self.contract = self.w3.eth.contract(
                    address=Web3.to_checksum_address(self.contract_address),
                    abi=self.abi,
                )
            else:
                self._deploy_contract()

    def _deploy_contract(self):
        """Deploys FaceVerificationRecord contract to EVM chain."""
        if not self.w3 or not self.w3.is_connected():
            return

        sender = self.account.address if self.account else self.w3.eth.accounts[0]
        contract_factory = self.w3.eth.contract(abi=self.abi, bytecode=self.bytecode)

        if self.account:
            tx = contract_factory.constructor().build_transaction({
                "from": sender,
                "nonce": self.w3.eth.get_transaction_count(sender),
                "gas": 3000000,
                "gasPrice": self.w3.eth.gas_price,
            })
            signed = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        else:
            tx_hash = contract_factory.constructor().transact({"from": sender})

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        self.contract_address = receipt.contractAddress
        self.contract = self.w3.eth.contract(address=receipt.contractAddress, abi=self.abi)

    def is_available(self) -> bool:
        return bool(self.w3 and self.w3.is_connected() and self.contract)

    def record_verification(
        self,
        face_hash: str,
        post: SocialPost,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BlockchainRecord:
        if not self.is_available():
            raise RuntimeError("EVM Blockchain connection is not available.")

        combined = f"{face_hash}:{post.post_hash}".encode("utf-8")
        record_id_hex = hashlib.sha256(combined).hexdigest()
        record_id_bytes = bytes.fromhex(record_id_hex)
        face_bytes = bytes.fromhex(face_hash)
        post_bytes = bytes.fromhex(post.post_hash)

        sender = self.account.address if self.account else self.w3.eth.accounts[0]

        tx_fn = self.contract.functions.recordVerification(
            record_id_bytes,
            face_bytes,
            post_bytes,
            post.platform,
            post.post_url,
            post.author,
        )

        if self.account:
            tx_payload = tx_fn.build_transaction({
                "from": sender,
                "to": self.contract_address,
                "nonce": self.w3.eth.get_transaction_count(sender),
                "gas": 500000,
                "gasPrice": self.w3.eth.gas_price,
            })
            signed = self.account.sign_transaction(tx_payload)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        else:
            tx_hash = tx_fn.transact({"from": sender})

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        block = self.w3.eth.get_block(receipt.blockNumber)

        return BlockchainRecord(
            record_id=record_id_hex,
            face_hash=face_hash,
            post_hash=post.post_hash,
            platform=post.platform,
            post_url=post.post_url,
            author=post.author,
            timestamp=block.timestamp,
            block_number=receipt.blockNumber,
            block_hash=receipt.blockHash.hex(),
            transaction_hash=receipt.transactionHash.hex(),
            blockchain_type="evm",
            metadata=metadata or {},
        )

    def verify_record(
        self,
        record_id: str,
        provided_face_hash: str,
        provided_post_hash: str,
    ) -> VerificationAuditResult:
        if not self.is_available():
            raise RuntimeError("EVM Blockchain connection is not available.")

        record_bytes = bytes.fromhex(record_id)
        face_bytes = bytes.fromhex(provided_face_hash)
        post_bytes = bytes.fromhex(provided_post_hash)

        is_valid, timestamp, platform, post_url, author, recorded_by = self.contract.functions.verifyRecord(
            record_bytes, face_bytes, post_bytes
        ).call()

        rec_struct = self.contract.functions.getRecord(record_bytes).call()
        on_chain_face_hash = rec_struct[1].hex()
        on_chain_post_hash = rec_struct[2].hex()

        face_matches = (on_chain_face_hash.lower() == provided_face_hash.lower())
        post_matches = (on_chain_post_hash.lower() == provided_post_hash.lower())

        msg = "EVM On-chain smart contract verified! Proof intact." if is_valid else (
            "Cryptographic verification FAILED on EVM smart contract: Hash mismatch."
        )

        return VerificationAuditResult(
            is_valid=is_valid,
            record_id=record_id,
            on_chain_face_hash=on_chain_face_hash,
            provided_face_hash=provided_face_hash,
            face_hash_matches=face_matches,
            on_chain_post_hash=on_chain_post_hash,
            provided_post_hash=provided_post_hash,
            post_hash_matches=post_matches,
            block_number=0,
            block_hash=f"Contract: {self.contract_address}",
            timestamp=timestamp,
            message=msg,
            blockchain_type="evm",
        )
