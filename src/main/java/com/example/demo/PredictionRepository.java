package com.example.demo;

import org.springframework.data.mongodb.repository.MongoRepository;

public interface PredictionRepository extends MongoRepository<BusData, String> {
}